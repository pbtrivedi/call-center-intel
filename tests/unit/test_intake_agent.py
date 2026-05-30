"""Tests for src/agents/intake_agent.py."""
from __future__ import annotations

import io
import uuid
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.intake_agent import IntakeAgent
from src.config.loader import Settings
from src.models.schemas import AudioInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        llm_provider="openai",
        whisper_model="base",
        max_file_size_mb=50,
        max_duration_minutes=60,
        db_path="data/calls.db",
        app_port=7860,
        max_temp_files=10,
        log_level="INFO",
        langsmith_project="test",
        langsmith_api_key="",
        openai_api_key="",
        gemini_api_key="",
        groq_api_key="",
        hf_token="",
    )
    return Settings(**{**defaults, **overrides})


def _wav_bytes(num_frames: int = 16000, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


def _mp3_bytes() -> bytes:
    return b"ID3" + b"\x03\x00\x00" + b"\x00" * 128


def _ogg_bytes() -> bytes:
    return b"OggS" + b"\x00" * 20


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    """IntakeAgent writing temp files into tmp_path."""
    monkeypatch.setattr("src.agents.intake_agent._TEMP_SUBDIR", str(tmp_path))
    return IntakeAgent(settings=_make_settings())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_wav_returns_valid_intake(tmp_path, agent):
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(_wav_bytes())
    result = agent.run(AudioInput(file_path=str(audio_file), filename="call.wav"))
    assert result.valid is True
    assert result.audio_properties is not None
    assert result.audio_properties.format == "wav"
    assert result.temp_file_path is not None


def test_valid_mp3_returns_valid_intake(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agents.intake_agent._TEMP_SUBDIR", str(tmp_path))
    agent = IntakeAgent(settings=_make_settings())

    audio_file = tmp_path / "call.mp3"
    audio_file.write_bytes(_mp3_bytes())

    # mutagen can't read our stub MP3 bytes, so mock get_duration
    with patch("src.agents.intake_agent.get_duration", return_value=30.0):
        result = agent.run(AudioInput(file_path=str(audio_file), filename="call.mp3"))

    assert result.valid is True
    assert result.audio_properties.format == "mp3"
    assert result.audio_properties.duration_seconds == pytest.approx(30.0)


def test_sha256_present_and_64_chars(tmp_path, agent):
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(_wav_bytes())
    result = agent.run(AudioInput(file_path=str(audio_file), filename="call.wav"))
    assert result.audio_properties is not None
    h = result.audio_properties.sha256_hash
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_call_id_is_valid_uuid(tmp_path, agent):
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(_wav_bytes())
    result = agent.run(AudioInput(file_path=str(audio_file), filename="call.wav"))
    uuid.UUID(result.call_id)  # raises ValueError if not a valid UUID


def test_temp_file_created_on_disk(tmp_path, agent):
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(_wav_bytes())
    result = agent.run(AudioInput(file_path=str(audio_file), filename="call.wav"))
    assert result.temp_file_path is not None
    assert Path(result.temp_file_path).exists()


def test_file_size_bytes_matches(tmp_path, agent):
    data = _wav_bytes()
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(data)
    result = agent.run(AudioInput(file_path=str(audio_file), filename="call.wav"))
    assert result.audio_properties.file_size_bytes == len(data)


# ---------------------------------------------------------------------------
# Format rejection
# ---------------------------------------------------------------------------


def test_unsupported_format_rejected(tmp_path, agent):
    audio_file = tmp_path / "call.ogg"
    audio_file.write_bytes(_ogg_bytes())
    result = agent.run(AudioInput(file_path=str(audio_file), filename="call.ogg"))
    assert result.valid is False
    assert result.validation_error is not None
    for fmt in ("wav", "mp3", "flac", "m4a"):
        assert fmt in result.validation_error


def test_wav_extension_mp3_content_accepted_as_mp3(tmp_path, monkeypatch):
    """Magic bytes override file extension."""
    monkeypatch.setattr("src.agents.intake_agent._TEMP_SUBDIR", str(tmp_path))
    agent = IntakeAgent(settings=_make_settings())
    audio_file = tmp_path / "call.wav"   # .wav extension
    audio_file.write_bytes(_mp3_bytes())  # MP3 magic bytes

    with patch("src.agents.intake_agent.get_duration", return_value=10.0):
        result = agent.run(AudioInput(file_path=str(audio_file), filename="call.wav"))

    assert result.valid is True
    assert result.audio_properties.format == "mp3"


# ---------------------------------------------------------------------------
# Size gate
# ---------------------------------------------------------------------------


def test_file_too_large_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agents.intake_agent._TEMP_SUBDIR", str(tmp_path))
    agent = IntakeAgent(settings=_make_settings(max_file_size_mb=1))

    audio_file = tmp_path / "big.wav"
    # Build a WAV header then pad to exceed 1 MB
    header = _wav_bytes(num_frames=100)
    audio_file.write_bytes(header + b"\x00" * (1024 * 1024 + 1))

    result = agent.run(AudioInput(file_path=str(audio_file), filename="big.wav"))
    assert result.valid is False
    assert "exceeds limit" in result.validation_error


# ---------------------------------------------------------------------------
# Duration gate
# ---------------------------------------------------------------------------


def test_duration_too_long_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agents.intake_agent._TEMP_SUBDIR", str(tmp_path))
    agent = IntakeAgent(settings=_make_settings(max_duration_minutes=1))

    audio_file = tmp_path / "long.wav"
    # 90 seconds at 16 kHz → exceeds 1-minute limit
    audio_file.write_bytes(_wav_bytes(num_frames=90 * 16000, sample_rate=16000))
    result = agent.run(AudioInput(file_path=str(audio_file), filename="long.wav"))
    assert result.valid is False
    assert "exceeds limit" in result.validation_error


def test_duration_at_limit_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agents.intake_agent._TEMP_SUBDIR", str(tmp_path))
    agent = IntakeAgent(settings=_make_settings(max_duration_minutes=1))

    audio_file = tmp_path / "ok.wav"
    # Exactly 60 seconds
    audio_file.write_bytes(_wav_bytes(num_frames=60 * 16000, sample_rate=16000))
    result = agent.run(AudioInput(file_path=str(audio_file), filename="ok.wav"))
    assert result.valid is True


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


def test_missing_file_rejected(tmp_path, agent):
    result = agent.run(AudioInput(file_path="/nonexistent/path/call.wav", filename="call.wav"))
    assert result.valid is False
    assert result.validation_error is not None


# ---------------------------------------------------------------------------
# Metadata PII scan (warns but does not reject)
# ---------------------------------------------------------------------------


def test_metadata_pii_does_not_reject(tmp_path, agent):
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(_wav_bytes())
    with patch.object(agent._logger, "warning") as mock_warn:
        result = agent.run(AudioInput(
            file_path=str(audio_file),
            filename="call.wav",
            caller_id="123-45-6789",  # SSN pattern
        ))
    assert result.valid is True  # PII in metadata does not reject the call
    pii_calls = [c for c in mock_warn.call_args_list if "PII" in str(c)]
    assert pii_calls, "Expected a PII warning to be logged"
