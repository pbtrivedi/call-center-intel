"""Tests for src/services/audio_utils.py."""
from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from src.common.exceptions import AudioValidationError
from src.services.audio_utils import (
    cleanup_temp_files,
    compute_sha256,
    detect_format,
    get_duration,
)


# ---------------------------------------------------------------------------
# Helpers: minimal valid audio byte sequences
# ---------------------------------------------------------------------------


def _wav_bytes(num_frames: int = 16000, sample_rate: int = 16000) -> bytes:
    """Return a minimal single-channel 16-bit PCM WAV in memory."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


def _mp3_id3_bytes() -> bytes:
    return b"ID3" + b"\x03\x00\x00" + b"\x00" * 10


def _mp3_sync_bytes() -> bytes:
    # MPEG-1 layer 3, 128 kbps, 44100 Hz sync word
    return b"\xff\xfb\x90\x00" + b"\x00" * 100


def _flac_bytes() -> bytes:
    return b"fLaC" + b"\x00" * 30


def _m4a_bytes() -> bytes:
    # ISO Base Media: 4-byte size + "ftyp" + brand
    return b"\x00\x00\x00\x1c" + b"ftyp" + b"M4A " + b"\x00" * 12


def _garbage_bytes() -> bytes:
    return b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c"


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


def test_detect_wav():
    assert detect_format(_wav_bytes()) == "wav"


def test_detect_mp3_id3():
    assert detect_format(_mp3_id3_bytes()) == "mp3"


def test_detect_mp3_sync():
    assert detect_format(_mp3_sync_bytes()) == "mp3"


def test_detect_flac():
    assert detect_format(_flac_bytes()) == "flac"


def test_detect_m4a():
    assert detect_format(_m4a_bytes()) == "m4a"


def test_detect_ignores_extension():
    """Content wins over filename — MP3 magic bytes in a .wav file → 'mp3'."""
    assert detect_format(_mp3_id3_bytes()) == "mp3"


def test_detect_unsupported_raises():
    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        detect_format(_garbage_bytes())


def test_detect_unsupported_lists_formats():
    with pytest.raises(AudioValidationError) as exc_info:
        detect_format(_garbage_bytes())
    msg = str(exc_info.value)
    for fmt in ("wav", "mp3", "flac", "m4a"):
        assert fmt in msg


def test_detect_too_small_raises():
    with pytest.raises(AudioValidationError, match="too small"):
        detect_format(b"\x00\x01")


# ---------------------------------------------------------------------------
# compute_sha256
# ---------------------------------------------------------------------------


def test_sha256_is_64_hex_chars():
    digest = compute_sha256(b"hello")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_sha256_same_input_same_output():
    data = b"call center audio"
    assert compute_sha256(data) == compute_sha256(data)


def test_sha256_different_inputs_differ():
    assert compute_sha256(b"aaa") != compute_sha256(b"bbb")


def test_sha256_known_value():
    # echo -n '' | sha256sum → e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    assert compute_sha256(b"") == "e3b0c44298fc1c149afbf4c8996fb924" \
                                   "27ae41e4649b934ca495991b7852b855"


# ---------------------------------------------------------------------------
# get_duration (WAV path via wave module)
# ---------------------------------------------------------------------------


def test_get_wav_duration_one_second(tmp_path):
    path = tmp_path / "test.wav"
    path.write_bytes(_wav_bytes(num_frames=16000, sample_rate=16000))
    assert get_duration(path, "wav") == pytest.approx(1.0, abs=0.01)


def test_get_wav_duration_two_seconds(tmp_path):
    path = tmp_path / "test.wav"
    path.write_bytes(_wav_bytes(num_frames=32000, sample_rate=16000))
    assert get_duration(path, "wav") == pytest.approx(2.0, abs=0.01)


def test_get_wav_duration_invalid_file_raises(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wav file at all")
    with pytest.raises(AudioValidationError):
        get_duration(path, "wav")


# ---------------------------------------------------------------------------
# cleanup_temp_files
# ---------------------------------------------------------------------------


def test_cleanup_removes_oldest(tmp_path):
    # Create 5 files with distinct mtimes
    files = []
    for i in range(5):
        f = tmp_path / f"file_{i}.wav"
        f.write_bytes(b"x")
        files.append(f)
        # Ensure strictly increasing mtime by touching each
        import os, time
        os.utime(f, (time.time() + i, time.time() + i))

    removed = cleanup_temp_files(tmp_path, max_files=3)
    assert removed == 2
    remaining = list(tmp_path.iterdir())
    assert len(remaining) == 3


def test_cleanup_no_op_under_limit(tmp_path):
    for i in range(2):
        (tmp_path / f"f{i}.wav").write_bytes(b"x")
    removed = cleanup_temp_files(tmp_path, max_files=5)
    assert removed == 0
    assert len(list(tmp_path.iterdir())) == 2


def test_cleanup_empty_dir(tmp_path):
    assert cleanup_temp_files(tmp_path, max_files=3) == 0


def test_cleanup_exact_limit(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.wav").write_bytes(b"x")
    removed = cleanup_temp_files(tmp_path, max_files=3)
    assert removed == 0
