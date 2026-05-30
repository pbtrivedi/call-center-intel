from __future__ import annotations

import re
from collections.abc import Callable
from typing import NamedTuple

from faster_whisper import WhisperModel

from src.common.exceptions import TranscriptionError
from src.common.logger import get_logger
from src.models.schemas import IntakeResult, TranscriptionResult, TranscriptionSegment
from src.services.whisper_model import get_whisper_model

# ---------------------------------------------------------------------------
# Diarization constants
# ---------------------------------------------------------------------------

_GAP_THRESHOLD = 1.5  # seconds of silence before considering a speaker change
_MIN_CONFIDENCE = 0.05  # below this, the segment is very likely hallucinated by Whisper

# ---------------------------------------------------------------------------
# Artifact cleaning
# ---------------------------------------------------------------------------

_ARTIFACT_PATTERNS = [
    re.compile(r"\[BLANK_AUDIO\]", re.IGNORECASE),
    re.compile(r"\[Music\]", re.IGNORECASE),
    re.compile(r"\[Applause\]", re.IGNORECASE),
    re.compile(r"\(Blank Audio\)", re.IGNORECASE),
    re.compile(r"\bsubscribe\b.{0,40}channel\b.*", re.IGNORECASE),
    re.compile(r"\bthanks? for watching\b.*", re.IGNORECASE),
    re.compile(r"\blike and (share|subscribe)\b.*", re.IGNORECASE),
]
# Collapses runs of identical phrases: "Thank you. Thank you." → "Thank you."
_REPEAT_RE = re.compile(r"(\b\w[\w ',\-]{2,}\b)([.!?]?\s+\1)+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal raw segment from faster-whisper
# ---------------------------------------------------------------------------


class _RawSegment(NamedTuple):
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


# ---------------------------------------------------------------------------
# TranscriptionAgent
# ---------------------------------------------------------------------------


class TranscriptionAgent:
    def __init__(
        self,
        model_factory: Callable[[], WhisperModel] = get_whisper_model,
        cache: dict[str, TranscriptionResult] | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._cache: dict[str, TranscriptionResult] = {} if cache is None else cache
        self._logger = get_logger(__name__)

    def run(self, intake_result: IntakeResult) -> TranscriptionResult:
        if not intake_result.valid or intake_result.audio_properties is None:
            raise TranscriptionError(
                "Cannot transcribe an invalid intake result",
                context={"call_id": intake_result.call_id},
            )

        sha256 = intake_result.audio_properties.sha256_hash
        call_id = intake_result.call_id

        # --- cache hit -------------------------------------------------------
        if sha256 in self._cache:
            cached = self._cache[sha256]
            self._logger.info("transcription cache_hit call_id=%s sha256=%s...", call_id, sha256[:8])
            # Return a copy with the current call_id and from_cache=True
            return TranscriptionResult(
                call_id=call_id,
                full_text=cached.full_text,
                segments=list(cached.segments),
                language=cached.language,
                duration_seconds=cached.duration_seconds,
                sha256_hash=sha256,
                from_cache=True,
            )

        # --- transcribe ------------------------------------------------------
        self._logger.info("transcription start call_id=%s sha256=%s...", call_id, sha256[:8])
        model = self._model_factory()

        try:
            segments_iter, info = model.transcribe(
                intake_result.temp_file_path,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            raw_segments = [
                _RawSegment(
                    start=s.start,
                    end=s.end,
                    text=s.text,
                    avg_logprob=s.avg_logprob,
                    no_speech_prob=s.no_speech_prob,
                )
                for s in segments_iter
            ]
        except Exception as e:
            raise TranscriptionError(
                f"Whisper transcription failed: {e}",
                context={"call_id": call_id},
            ) from e

        # --- speaker diarization ---------------------------------------------
        # Try pyannote (voice-fingerprint based) first; fall back to gap-based.
        from src.services.diarization import diarize
        diarization_segments = diarize(intake_result.temp_file_path)
        if diarization_segments is not None:
            speakers = _assign_speakers_from_diarization(raw_segments, diarization_segments)
            self._logger.info("using pyannote diarization call_id=%s", call_id)
        else:
            speakers = _assign_speaker_ids(raw_segments)
            self._logger.info("using gap-based speaker assignment call_id=%s", call_id)

        # --- clean + build segments ------------------------------------------
        final_segments: list[TranscriptionSegment] = []

        for raw, speaker in zip(raw_segments, speakers):
            cleaned = _clean_text(raw.text)
            if not cleaned:
                continue
            confidence = _compute_confidence(raw.avg_logprob, raw.no_speech_prob)
            if confidence < _MIN_CONFIDENCE:
                self._logger.debug(
                    "dropping low-confidence segment start=%.1fs conf=%.4f text=%r",
                    raw.start, confidence, cleaned[:60],
                )
                continue
            final_segments.append(
                TranscriptionSegment(
                    speaker=speaker,
                    text=cleaned,
                    start_time=round(raw.start, 3),
                    end_time=round(raw.end, 3),
                    confidence=_compute_confidence(raw.avg_logprob, raw.no_speech_prob),
                )
            )

        full_text = " ".join(s.text for s in final_segments)
        duration = getattr(info, "duration", intake_result.audio_properties.duration_seconds)

        result = TranscriptionResult(
            call_id=call_id,
            full_text=full_text,
            segments=final_segments,
            language=getattr(info, "language", "en"),
            duration_seconds=round(duration, 3),
            sha256_hash=sha256,
            from_cache=False,
        )

        self._cache[sha256] = result
        self._logger.info(
            "transcription done call_id=%s segments=%d language=%s",
            call_id, len(final_segments), result.language,
        )
        return result


# ---------------------------------------------------------------------------
# Pure helper functions (testable independently)
# ---------------------------------------------------------------------------


def _compute_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    """Map faster-whisper segment quality metrics to a 0.0–1.0 confidence score."""
    logprob_score = max(0.0, min(1.0, 1.0 + avg_logprob))  # [-1, 0] → [0, 1]
    confidence = logprob_score * (1.0 - no_speech_prob)
    return round(max(0.0, min(1.0, confidence)), 4)


def _clean_text(text: str) -> str:
    """Remove Whisper artifacts and collapse repeated phrases."""
    for pattern in _ARTIFACT_PATTERNS:
        text = pattern.sub("", text)
    text = _REPEAT_RE.sub(r"\1", text)
    return text.strip()


def _assign_speaker_ids(segments: list[_RawSegment]) -> list[str]:
    """Fallback: assign neutral speaker IDs based on silence gaps only.

    Used when pyannote diarization is unavailable (no HF_TOKEN or import error).
    Toggles between SPEAKER_1 and SPEAKER_2 on each gap > _GAP_THRESHOLD.
    """
    if not segments:
        return []

    speaker_idx = 0  # 0 → SPEAKER_1, 1 → SPEAKER_2
    prev_end = 0.0
    result: list[str] = []

    for i, seg in enumerate(segments):
        if i > 0 and (seg.start - prev_end) > _GAP_THRESHOLD:
            speaker_idx = 1 - speaker_idx
        result.append(f"SPEAKER_{speaker_idx + 1}")
        prev_end = seg.end

    return result


def _assign_speakers_from_diarization(
    whisper_segments: list[_RawSegment],
    diarization_segments: list,
) -> list[str]:
    """Align pyannote diarization output to Whisper segments.

    For each Whisper segment, find the diarization segment with the greatest
    time overlap and use its speaker label. Falls back to 'SPEAKER_UNKNOWN'
    if no overlap is found (e.g., silence regions).
    """
    result: list[str] = []
    for wseg in whisper_segments:
        best_speaker = "SPEAKER_UNKNOWN"
        best_overlap = 0.0
        for dseg in diarization_segments:
            overlap = min(wseg.end, dseg.end) - max(wseg.start, dseg.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg.speaker
        result.append(best_speaker)
    return result
