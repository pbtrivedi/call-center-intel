import logging
import logging.handlers
import os
import threading
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = Path(__file__).parent.parent.parent / "logs"

_configured: set[str] = set()
_configured_lock = threading.Lock()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    with _configured_lock:
        if name in _configured:
            return logger

        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)
        logger.propagate = False

        formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_DIR / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        _configured.add(name)
    return logger
