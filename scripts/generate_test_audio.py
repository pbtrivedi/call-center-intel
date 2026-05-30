"""
Generate a synthetic ~45-second account-unlock call using OpenAI TTS.

Two voices:
  nova  → Sarah (agent) — warm, professional
  onyx  → James (customer) — frustrated then relieved

Includes PII (SSN, DOB) in the verification exchange to exercise the
pipeline's PII-redaction agent.

Usage:
    conda activate call-center-intel
    python scripts/generate_test_audio.py

Output:
    data/calls_audio/synthetic_account_unlock.mp3

Dependencies (already in the project env):
    openai, pydub
    ffmpeg must be on PATH: brew install ffmpeg
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from pydub import AudioSegment

client = OpenAI()

# ---------------------------------------------------------------------------
# Script  (speaker, openai-voice, line)
# ---------------------------------------------------------------------------
# Target: ~110 spoken words → ~45 s at 130 wpm + 450 ms pauses between turns
# Includes SSN and DOB in the verification step to test PII redaction.
# ---------------------------------------------------------------------------

SCRIPT: list[tuple[str, str, str]] = [
    (
        "customer", "onyx",
        "Hi, my account is locked and I have a payment due today.",
    ),
    (
        "agent", "nova",
        "I can take care of that right away. My name is Sarah — "
        "could I get your full name and date of birth to verify your identity?",
    ),
    (
        "customer", "onyx",
        "Sure. James Miller, date of birth March 15, 1985.",
    ),
    (
        "agent", "nova",
        "Thank you, James. And for security I'll need the last four digits "
        "of your Social Security Number.",
    ),
    (
        "customer", "onyx",
        "It's 6789. My full SSN is 523-41-6789 if that helps.",
    ),
    (
        "agent", "nova",
        "Perfect, that matches what we have on file. Your account was locked "
        "after three failed login attempts this morning. I've unlocked it — "
        "you can sign in right now. I'm also sending a password reset link "
        "to your email so this doesn't happen again.",
    ),
    (
        "customer", "onyx",
        "That's great, thank you. Will my payment go through today?",
    ),
    (
        "agent", "nova",
        "Absolutely — your account is fully active. The reset link will arrive "
        "within the next minute. Is there anything else I can help you with?",
    ),
    (
        "customer", "onyx",
        "No, that's everything. Thank you so much, you've been really helpful.",
    ),
    (
        "agent", "nova",
        "My pleasure, James. Have a wonderful rest of your day!",
    ),
]

PAUSE_MS = 1800  # silence between speaker turns — must exceed _GAP_THRESHOLD (1.5 s) in transcription_agent.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tts_segment(text: str, voice: str) -> AudioSegment:
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="mp3",
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(response.content)
        tmp_path = f.name
    try:
        return AudioSegment.from_mp3(tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path("data/calls_audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_account_unlock.mp3"

    pause = AudioSegment.silent(duration=PAUSE_MS)
    combined = AudioSegment.empty()

    print("Generating TTS segments…\n")
    for i, (speaker, voice, text) in enumerate(SCRIPT):
        preview = text[:72] + ("…" if len(text) > 72 else "")
        print(f"  [{speaker:8s}] {preview}")
        segment = _tts_segment(text, voice)
        if i > 0:
            combined += pause
        combined += segment

    combined.export(str(out_path), format="mp3")
    duration_s = len(combined) / 1000
    print(f"\n  Saved -> {out_path}  ({duration_s:.1f} s)")


if __name__ == "__main__":
    main()
