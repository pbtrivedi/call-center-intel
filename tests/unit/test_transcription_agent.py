"""Tests for src/agents/transcription_agent.py.

The Whisper model is NEVER loaded — all tests inject a mock via model_factory.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agents.transcription_agent import (
    TranscriptionAgent,
    _assign_speaker_ids,
    _assign_speakers_from_diarization,
    _clean_text,
    _compute_confidence,
    _RawSegment,
)
from src.common.exceptions import TranscriptionError
from src.models.schemas import AudioProperties, IntakeResult, TranscriptionResult

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_SHA256 = "a" * 64


def _make_intake(sha256: str = _SHA256, temp_path: str = "/tmp/call.wav") -> IntakeResult:
    props = AudioProperties(
        format="wav",
        duration_seconds=10.0,
        file_size_bytes=160_000,
        sha256_hash=sha256,
    )
    return IntakeResult(
        valid=True,
        call_id="call-test-1",
        audio_properties=props,
        temp_file_path=temp_path,
    )


def _make_raw(
    text: str,
    start: float = 0.0,
    end: float = 1.0,
    avg_logprob: float = -0.2,
    no_speech_prob: float = 0.05,
) -> _RawSegment:
    return _RawSegment(
        start=start,
        end=end,
        text=text,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


def _make_mock_model(segments: list[_RawSegment], language: str = "en") -> MagicMock:
    """Build a mock WhisperModel whose transcribe() returns the given segments."""
    info = SimpleNamespace(language=language, duration=sum(s.end for s in segments[-1:]) if segments else 0.0)

    mock_segs = []
    for s in segments:
        ms = SimpleNamespace(
            start=s.start,
            end=s.end,
            text=s.text,
            avg_logprob=s.avg_logprob,
            no_speech_prob=s.no_speech_prob,
        )
        mock_segs.append(ms)

    model = MagicMock()
    model.transcribe.return_value = (iter(mock_segs), info)
    return model


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------


def test_confidence_perfect_signal():
    # avg_logprob=0.0 maps to 1.0; no_speech_prob=0.0 → no penalty
    assert _compute_confidence(0.0, 0.0) == pytest.approx(1.0)


def test_confidence_high_noise():
    assert _compute_confidence(-1.5, 0.9) == pytest.approx(0.0, abs=0.05)


def test_confidence_clamped_to_zero():
    assert _compute_confidence(-2.0, 1.0) == pytest.approx(0.0)


def test_confidence_clamped_to_one():
    assert _compute_confidence(0.5, 0.0) == pytest.approx(1.0)


def test_confidence_typical_good_segment():
    # avg_logprob=-0.2 → logprob_score=0.8; no_speech_prob=0.05 → 0.8*0.95=0.76
    assert _compute_confidence(-0.2, 0.05) == pytest.approx(0.76, abs=0.01)


def test_confidence_in_unit_range():
    for logprob in (-2.0, -1.0, -0.5, -0.1, 0.0):
        for nsp in (0.0, 0.3, 0.7, 1.0):
            c = _compute_confidence(logprob, nsp)
            assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------


def test_clean_removes_blank_audio():
    assert _clean_text("[BLANK_AUDIO]") == ""


def test_clean_removes_music_marker():
    assert _clean_text("[Music]") == ""


def test_clean_removes_applause():
    assert _clean_text("[Applause]") == ""


def test_clean_removes_blank_audio_paren():
    assert _clean_text("(Blank Audio)") == ""


def test_clean_collapses_repeated_phrase():
    result = _clean_text("Thank you. Thank you. Thank you.")
    assert result.count("Thank you") == 1


def test_clean_strips_whitespace():
    assert _clean_text("  hello  ") == "hello"


def test_clean_preserves_normal_text():
    text = "Hello, how can I help you today?"
    assert _clean_text(text) == text


def test_clean_removes_youtube_footer():
    assert _clean_text("Thanks for watching our channel") == ""


# ---------------------------------------------------------------------------
# _assign_speaker_ids  (gap-based fallback)
# ---------------------------------------------------------------------------


def test_single_segment_is_speaker_1():
    segs = [_make_raw("Hello.", start=0.0, end=2.0)]
    assert _assign_speaker_ids(segs) == ["SPEAKER_1"]


def test_empty_segments_returns_empty():
    assert _assign_speaker_ids([]) == []


def test_gap_above_threshold_toggles_speaker():
    segs = [
        _make_raw("Hello.", start=0.0, end=2.0),
        _make_raw("Hi there.", start=4.0, end=6.0),   # 2 s gap > 1.5 threshold
    ]
    assert _assign_speaker_ids(segs) == ["SPEAKER_1", "SPEAKER_2"]


def test_gap_below_threshold_keeps_same_speaker():
    segs = [
        _make_raw("Let me check that.", start=0.0, end=2.0),
        _make_raw("One moment please.", start=2.1, end=4.0),  # 0.1 s gap
    ]
    assert _assign_speaker_ids(segs) == ["SPEAKER_1", "SPEAKER_1"]


def test_gap_exactly_at_threshold_does_not_toggle():
    segs = [
        _make_raw("First.", start=0.0, end=2.0),
        _make_raw("Second.", start=3.5, end=5.0),   # gap == 1.5 s exactly
    ]
    assert _assign_speaker_ids(segs) == ["SPEAKER_1", "SPEAKER_1"]


def test_three_turns_alternates_correctly():
    segs = [
        _make_raw("Turn A.", start=0.0, end=2.0),
        _make_raw("Turn B.", start=4.0, end=6.0),
        _make_raw("Turn A again.", start=8.0, end=10.0),
    ]
    assert _assign_speaker_ids(segs) == ["SPEAKER_1", "SPEAKER_2", "SPEAKER_1"]


def test_content_does_not_affect_gap_assignment():
    # Customer/agent phrases must have no effect — pure gap logic only
    segs = [
        _make_raw("I need help with my account.", start=0.0, end=2.0),
        _make_raw("Let me check that for you.", start=2.1, end=4.0),  # tiny gap
    ]
    assert _assign_speaker_ids(segs) == ["SPEAKER_1", "SPEAKER_1"]


# ---------------------------------------------------------------------------
# _assign_speakers_from_diarization
# ---------------------------------------------------------------------------


def test_diarization_assigns_correct_speaker():
    from src.services.diarization import DiarizationSegment
    from src.agents.transcription_agent import _assign_speakers_from_diarization

    whisper = [
        _make_raw("Hello.", start=0.0, end=2.0),
        _make_raw("Hi there.", start=3.0, end=5.0),
    ]
    diar = [
        DiarizationSegment(start=0.0, end=2.5, speaker="SPEAKER_00"),
        DiarizationSegment(start=2.8, end=5.5, speaker="SPEAKER_01"),
    ]
    result = _assign_speakers_from_diarization(whisper, diar)
    assert result == ["SPEAKER_00", "SPEAKER_01"]


def test_diarization_unknown_when_no_overlap():
    from src.services.diarization import DiarizationSegment
    from src.agents.transcription_agent import _assign_speakers_from_diarization

    whisper = [_make_raw("Silence region.", start=10.0, end=12.0)]
    diar = [DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00")]
    result = _assign_speakers_from_diarization(whisper, diar)
    assert result == ["SPEAKER_UNKNOWN"]


# ---------------------------------------------------------------------------
# TranscriptionAgent.run()
# ---------------------------------------------------------------------------


def test_returns_transcription_result():
    segs = [_make_raw("Hello, how can I help you today?", start=0.0, end=2.0)]
    model = _make_mock_model(segs)
    agent = TranscriptionAgent(model_factory=lambda: model)
    result = agent.run(_make_intake())
    assert isinstance(result, TranscriptionResult)
    assert result.call_id == "call-test-1"
    assert result.from_cache is False
    assert len(result.segments) == 1


def test_cache_miss_calls_model_once():
    segs = [_make_raw("Hello.", start=0.0, end=1.0)]
    model = _make_mock_model(segs)
    agent = TranscriptionAgent(model_factory=lambda: model)
    agent.run(_make_intake())
    model.transcribe.assert_called_once()


def test_cache_hit_skips_model():
    segs = [_make_raw("Hello.", start=0.0, end=1.0)]
    model = _make_mock_model(segs)
    agent = TranscriptionAgent(model_factory=lambda: model)
    intake = _make_intake()

    agent.run(intake)                    # cache miss — model called
    result2 = agent.run(intake)          # cache hit  — model not called again

    assert model.transcribe.call_count == 1
    assert result2.from_cache is True


def test_cache_hit_updates_call_id():
    segs = [_make_raw("Hi there.", start=0.0, end=1.0)]
    model = _make_mock_model(segs)
    agent = TranscriptionAgent(model_factory=lambda: model)

    agent.run(_make_intake(sha256=_SHA256))

    intake2 = IntakeResult(
        valid=True,
        call_id="call-different-id",
        audio_properties=AudioProperties(
            format="wav", duration_seconds=10.0,
            file_size_bytes=1000, sha256_hash=_SHA256,
        ),
        temp_file_path="/tmp/x.wav",
    )
    result2 = agent.run(intake2)
    assert result2.call_id == "call-different-id"
    assert result2.from_cache is True


def test_blank_audio_segment_dropped():
    segs = [
        _make_raw("[BLANK_AUDIO]", start=0.0, end=1.0),
        _make_raw("Can I help you?", start=1.5, end=3.0),
    ]
    model = _make_mock_model(segs)
    agent = TranscriptionAgent(model_factory=lambda: model)
    result = agent.run(_make_intake())
    assert len(result.segments) == 1
    assert "[BLANK_AUDIO]" not in result.full_text


def test_full_text_joins_cleaned_segments():
    segs = [
        _make_raw("Hello.", start=0.0, end=1.0),
        _make_raw("[BLANK_AUDIO]", start=1.1, end=2.0),
        _make_raw("How are you?", start=2.5, end=4.0),
    ]
    model = _make_mock_model(segs)
    agent = TranscriptionAgent(model_factory=lambda: model)
    result = agent.run(_make_intake())
    assert result.full_text == "Hello. How are you?"


def test_language_from_whisper_info():
    segs = [_make_raw("Hola.", start=0.0, end=1.0)]
    model = _make_mock_model(segs, language="es")
    agent = TranscriptionAgent(model_factory=lambda: model)
    result = agent.run(_make_intake())
    assert result.language == "es"


def test_invalid_intake_raises():
    agent = TranscriptionAgent(model_factory=MagicMock())
    invalid = IntakeResult(valid=False, call_id="x", validation_error="bad format")
    with pytest.raises(TranscriptionError):
        agent.run(invalid)


def test_segment_confidence_range():
    segs = [_make_raw("Test.", start=0.0, end=1.0, avg_logprob=-0.3, no_speech_prob=0.1)]
    model = _make_mock_model(segs)
    agent = TranscriptionAgent(model_factory=lambda: model)
    result = agent.run(_make_intake())
    c = result.segments[0].confidence
    assert 0.0 <= c <= 1.0


def test_external_cache_shared_between_agents():
    """Two agents sharing the same cache dict see each other's results."""
    shared_cache: dict = {}
    segs = [_make_raw("Shared content.", start=0.0, end=1.0)]
    model = _make_mock_model(segs)

    agent1 = TranscriptionAgent(model_factory=lambda: model, cache=shared_cache)
    agent2 = TranscriptionAgent(model_factory=MagicMock(), cache=shared_cache)

    agent1.run(_make_intake())
    result = agent2.run(_make_intake())

    assert result.from_cache is True


# ---------------------------------------------------------------------------
# Confidence filter — hallucination suppression
# ---------------------------------------------------------------------------


def test_low_confidence_segment_dropped():
    # avg_logprob=-2.0, no_speech_prob=0.9 → confidence ≈ 0.0
    segs = [
        _make_raw("Hallucinated noise text.", start=0.0, end=12.0, avg_logprob=-2.0, no_speech_prob=0.9),
        _make_raw("Hello, how can I help?", start=12.5, end=15.0, avg_logprob=-0.2, no_speech_prob=0.05),
    ]
    model = _make_mock_model(segs)
    agent = TranscriptionAgent(model_factory=lambda: model)
    result = agent.run(_make_intake())
    assert len(result.segments) == 1
    assert "Hallucinated" not in result.full_text


def test_borderline_confidence_segment_kept():
    # avg_logprob=-0.5, no_speech_prob=0.3 → confidence = 0.5 * 0.7 = 0.35 (well above 0.05)
    segs = [_make_raw("Can I help you?", start=0.0, end=2.0, avg_logprob=-0.5, no_speech_prob=0.3)]
    model = _make_mock_model(segs)
    agent = TranscriptionAgent(model_factory=lambda: model)
    result = agent.run(_make_intake())
    assert len(result.segments) == 1
