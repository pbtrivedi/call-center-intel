import pytest

import src.config.loader as loader_module
from src.common.exceptions import ConfigurationError
from src.config.loader import _reset_settings, get_settings


@pytest.fixture(autouse=True)
def reset_between_tests(monkeypatch):
    """Ensure each test starts with a clean settings cache and no .env side-effects."""
    monkeypatch.setattr("src.config.loader.load_dotenv", lambda **kw: None)
    _reset_settings()
    yield
    _reset_settings()


def test_default_llm_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert get_settings().llm_provider == "openai"


def test_llm_provider_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert get_settings().llm_provider == "groq"


def test_default_whisper_model(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert get_settings().whisper_model == "base"


def test_default_max_file_size(monkeypatch):
    monkeypatch.delenv("MAX_FILE_SIZE_MB", raising=False)
    assert get_settings().max_file_size_mb == 50


def test_default_max_duration(monkeypatch):
    monkeypatch.delenv("MAX_DURATION_MINUTES", raising=False)
    assert get_settings().max_duration_minutes == 60


def test_default_app_port(monkeypatch):
    monkeypatch.delenv("APP_PORT", raising=False)
    assert get_settings().app_port == 7860


def test_app_port_override(monkeypatch):
    monkeypatch.setenv("APP_PORT", "8080")
    assert get_settings().app_port == 8080


def test_default_max_temp_files(monkeypatch):
    monkeypatch.delenv("MAX_TEMP_FILES", raising=False)
    assert get_settings().max_temp_files == 10


def test_langsmith_api_key_defaults_empty(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert get_settings().langsmith_api_key == ""


def test_langsmith_project_default(monkeypatch):
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert get_settings().langsmith_project == "call-center-intel"


def test_settings_is_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_reset_clears_cache():
    s1 = get_settings()
    _reset_settings()
    s2 = get_settings()
    assert s1 is not s2


def test_invalid_llm_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gpt4")
    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        get_settings()


def test_invalid_int_env_var_raises(monkeypatch):
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "large")
    with pytest.raises(ConfigurationError, match="MAX_FILE_SIZE_MB"):
        get_settings()


def test_invalid_app_port_raises(monkeypatch):
    monkeypatch.setenv("APP_PORT", "eighty")
    with pytest.raises(ConfigurationError, match="APP_PORT"):
        get_settings()


def test_missing_settings_yaml_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(loader_module, "_SETTINGS_FILE", tmp_path / "nonexistent.yaml")
    with pytest.raises(ConfigurationError, match="not found"):
        get_settings()
