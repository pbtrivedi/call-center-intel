import logging
from pathlib import Path

import pytest

from src.common.logger import get_logger


def test_returns_logger_instance():
    logger = get_logger("test.instance")
    assert isinstance(logger, logging.Logger)


def test_logger_has_correct_name():
    logger = get_logger("test.naming")
    assert logger.name == "test.naming"


def test_logger_has_handlers():
    logger = get_logger("test.handlers")
    assert len(logger.handlers) >= 2  # stream + file


def test_logger_does_not_propagate():
    logger = get_logger("test.propagate")
    assert logger.propagate is False


def test_default_level_is_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logger = get_logger("test.default_info")
    assert logger.level == logging.INFO


def test_log_level_env_var_respected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    logger = get_logger("test.env_debug")
    assert logger.level == logging.DEBUG


def test_repeated_calls_same_name_do_not_duplicate_handlers():
    get_logger("test.dedup")
    logger = get_logger("test.dedup")
    # handlers should not double-up on the second call
    count_first = len(logging.getLogger("test.dedup").handlers)
    get_logger("test.dedup")
    assert len(logging.getLogger("test.dedup").handlers) == count_first


def test_logs_dir_created(tmp_path, monkeypatch):
    import src.common.logger as logger_module

    original_dir = logger_module._LOG_DIR
    logger_module._LOG_DIR = tmp_path / "logs"
    # Remove from configured set so it gets re-initialised
    logger_module._configured.discard("test.tmpdir")

    try:
        get_logger("test.tmpdir")
        assert (tmp_path / "logs").is_dir()
    finally:
        logger_module._LOG_DIR = original_dir
        logger_module._configured.discard("test.tmpdir")
