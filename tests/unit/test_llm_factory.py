"""Tests for src/services/llm_factory.py."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


def test_llm_factory_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mock_llm = MagicMock()
    with patch("langchain_openai.ChatOpenAI", return_value=mock_llm) as mock_cls:
        from importlib import reload
        import src.services.llm_factory as factory
        reload(factory)
        result = factory.get_llm()

    mock_cls.assert_called_once()
    assert result is mock_llm


def test_llm_factory_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")

    mock_llm = MagicMock()
    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm) as mock_cls:
        from importlib import reload
        import src.services.llm_factory as factory
        reload(factory)
        result = factory.get_llm()

    mock_cls.assert_called_once()
    assert result is mock_llm


def test_llm_factory_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gq-test")

    mock_llm = MagicMock()
    with patch("langchain_groq.ChatGroq", return_value=mock_llm) as mock_cls:
        from importlib import reload
        import src.services.llm_factory as factory
        reload(factory)
        result = factory.get_llm()

    mock_cls.assert_called_once()
    assert result is mock_llm


def test_llm_factory_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    from importlib import reload
    import src.services.llm_factory as factory
    reload(factory)
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        factory.get_llm()


def test_llm_factory_default_is_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mock_llm = MagicMock()
    with patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
        from importlib import reload
        import src.services.llm_factory as factory
        reload(factory)
        result = factory.get_llm()

    assert result is mock_llm


def test_llm_factory_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        from importlib import reload
        import src.services.llm_factory as factory
        reload(factory)
        factory.get_llm()

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4-turbo"
