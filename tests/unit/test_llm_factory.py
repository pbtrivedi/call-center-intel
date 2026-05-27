"""Tests for src/services/llm_factory.py."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.services.llm_factory import get_llm


def test_llm_factory_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mock_llm = MagicMock()
    with patch("langchain_openai.ChatOpenAI", return_value=mock_llm) as mock_cls:
        result = get_llm()

    mock_cls.assert_called_once()
    assert result is mock_llm


def test_llm_factory_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")

    mock_llm = MagicMock()
    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm) as mock_cls:
        result = get_llm()

    mock_cls.assert_called_once()
    assert result is mock_llm


def test_llm_factory_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gq-test")

    mock_llm = MagicMock()
    with patch("langchain_groq.ChatGroq", return_value=mock_llm) as mock_cls:
        result = get_llm()

    mock_cls.assert_called_once()
    assert result is mock_llm


def test_llm_factory_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm()


def test_llm_factory_default_is_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mock_llm = MagicMock()
    with patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
        result = get_llm()

    assert result is mock_llm


def test_llm_factory_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        get_llm()

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4-turbo"


def test_llm_factory_warns_on_missing_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("langchain_openai.ChatOpenAI", return_value=MagicMock()):
        with patch("src.services.llm_factory._logger") as mock_logger:
            get_llm()

    assert any(
        "OPENAI_API_KEY" in str(call)
        for call in mock_logger.warning.call_args_list
    )


def test_llm_factory_gemini_timeout_is_float(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")

    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
        get_llm()

    call_kwargs = mock_cls.call_args.kwargs
    assert isinstance(call_kwargs["timeout"], float)
    assert call_kwargs["timeout"] == 30.0
