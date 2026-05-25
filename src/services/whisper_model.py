from __future__ import annotations

import threading

from faster_whisper import WhisperModel

from src.common.logger import get_logger
from src.config.loader import get_settings

_model: WhisperModel | None = None
_model_lock = threading.Lock()
_logger = get_logger(__name__)


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore[import-untyped]
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_whisper_model() -> WhisperModel:
    """Return the process-wide WhisperModel singleton.

    Detects CUDA on first call and selects compute type accordingly.
    Thread-safe via double-checked locking.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        settings = get_settings()
        device = "cuda" if _cuda_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        _logger.info(
            "Loading Whisper model model=%s device=%s compute_type=%s",
            settings.whisper_model,
            device,
            compute_type,
        )
        _model = WhisperModel(
            settings.whisper_model,
            device=device,
            compute_type=compute_type,
        )
        _logger.info("Whisper model loaded")

    return _model


def _reset_whisper_model() -> None:
    """Reset the cached model. For use in tests only."""
    global _model
    _model = None
