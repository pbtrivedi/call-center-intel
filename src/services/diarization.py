"""
Speaker diarization service using pyannote/speaker-diarization-3.1.

Loads the model once and caches it as a module-level singleton.
Requires HF_TOKEN in the environment and model access accepted at:
  https://huggingface.co/pyannote/speaker-diarization-3.1

Falls back gracefully if the token is missing or pyannote is unavailable.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from src.common.logger import get_logger

_logger = get_logger(__name__)

_lock = threading.Lock()
_pipeline = None          # pyannote Pipeline, loaded on first use
_pipeline_failed = False  # set True if load fails so we don't retry every call


@dataclass(frozen=True)
class DiarizationSegment:
    start: float   # seconds
    end: float     # seconds
    speaker: str   # e.g. "SPEAKER_00", "SPEAKER_01"


def get_diarization_pipeline():
    """Return the cached pyannote pipeline, loading it on first call.

    Returns None if HF_TOKEN is missing or the model fails to load.
    """
    global _pipeline, _pipeline_failed

    if _pipeline is not None:
        return _pipeline
    if _pipeline_failed:
        return None

    with _lock:
        if _pipeline is not None:
            return _pipeline
        if _pipeline_failed:
            return None

        try:
            from src.config.loader import get_settings
            hf_token = get_settings().hf_token
            if not hf_token:
                _logger.warning(
                    "HF_TOKEN not set — speaker diarization disabled; "
                    "falling back to gap-based speaker assignment"
                )
                _pipeline_failed = True
                return None

            from pyannote.audio import Pipeline
            _logger.info("loading pyannote/speaker-diarization-3.1 …")
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            _logger.info("pyannote pipeline loaded")
        except Exception as exc:
            _logger.error("failed to load pyannote pipeline: %s", exc)
            _pipeline_failed = True
            return None

    return _pipeline


def diarize(audio_path: str) -> list[DiarizationSegment] | None:
    """Run speaker diarization on an audio file.

    Returns a list of DiarizationSegments ordered by start time,
    or None if diarization is unavailable.
    """
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        return None

    try:
        diarization = pipeline(audio_path)
    except Exception as exc:
        _logger.error("diarization failed for %s: %s", audio_path, exc)
        return None

    segments: list[DiarizationSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(DiarizationSegment(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
        ))

    segments.sort(key=lambda s: s.start)
    _logger.info(
        "diarization done path=%s segments=%d speakers=%s",
        audio_path,
        len(segments),
        sorted({s.speaker for s in segments}),
    )
    return segments


def _reset_diarization_pipeline() -> None:
    """Reset cached pipeline. For use in tests only."""
    global _pipeline, _pipeline_failed
    _pipeline = None
    _pipeline_failed = False
