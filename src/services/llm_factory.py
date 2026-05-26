from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from src.common.logger import get_logger

_logger = get_logger(__name__)


def get_llm() -> BaseChatModel:
    """
    Return a LangChain chat model driven entirely by env vars.

    LLM_PROVIDER  → openai | gemini | groq  (default: openai)
    Model name, API key, and timeout are read from provider-specific env vars.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        api_key = os.getenv("OPENAI_API_KEY")
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        _logger.info("llm provider=openai model=%s", model)
        return ChatOpenAI(model=model, api_key=api_key, timeout=timeout)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        api_key = os.getenv("GEMINI_API_KEY")
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        _logger.info("llm provider=gemini model=%s", model)
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, timeout=timeout)

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        api_key = os.getenv("GROQ_API_KEY")
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        _logger.info("llm provider=groq model=%s", model)
        return ChatGroq(model=model, api_key=api_key, timeout=timeout)

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Supported values: openai, gemini, groq"
    )
