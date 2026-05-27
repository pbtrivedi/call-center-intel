import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.common.exceptions import ConfigurationError

_SETTINGS_FILE = Path(__file__).parent / "settings.yaml"
_settings: "Settings | None" = None
_settings_lock = threading.Lock()
_VALID_LLM_PROVIDERS = {"openai", "groq", "gemini"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    whisper_model: str
    max_file_size_mb: int
    max_duration_minutes: int
    db_path: str
    app_port: int
    max_temp_files: int
    log_level: str
    langsmith_project: str
    langsmith_api_key: str = field(repr=False)
    openai_api_key: str = field(repr=False)
    gemini_api_key: str = field(repr=False)
    groq_api_key: str = field(repr=False)


def _parse_int(env_var: str, default: int) -> int:
    raw = os.getenv(env_var, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(
            f"Environment variable {env_var}={raw!r} is not a valid integer"
        )


def get_settings() -> Settings:
    global _settings
    if _settings is not None:
        return _settings
    with _settings_lock:
        if _settings is not None:
            return _settings

        load_dotenv()

        try:
            with open(_SETTINGS_FILE) as f:
                defaults = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError(
                f"settings.yaml not found at {_SETTINGS_FILE}",
                context={"path": str(_SETTINGS_FILE)},
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse settings.yaml: {e}",
                context={"path": str(_SETTINGS_FILE)},
            )

        llm_provider = os.getenv("LLM_PROVIDER", defaults["llm_provider"])
        if llm_provider not in _VALID_LLM_PROVIDERS:
            raise ConfigurationError(
                f"LLM_PROVIDER={llm_provider!r} is invalid; must be one of {_VALID_LLM_PROVIDERS}",
                context={"value": llm_provider, "valid": sorted(_VALID_LLM_PROVIDERS)},
            )

        _settings = Settings(
            llm_provider=llm_provider,
            whisper_model=os.getenv("WHISPER_MODEL", defaults["whisper_model"]),
            max_file_size_mb=_parse_int("MAX_FILE_SIZE_MB", defaults["max_file_size_mb"]),
            max_duration_minutes=_parse_int("MAX_DURATION_MINUTES", defaults["max_duration_minutes"]),
            db_path=os.getenv("DB_PATH", defaults["db_path"]),
            app_port=_parse_int("APP_PORT", defaults["app_port"]),
            max_temp_files=_parse_int("MAX_TEMP_FILES", defaults["max_temp_files"]),
            log_level=os.getenv("LOG_LEVEL", defaults["log_level"]),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", defaults["langsmith_project"]),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
        )
    return _settings


def get_langsmith_status() -> dict:
    """Return LangSmith tracing status for the Observability tab."""
    settings = get_settings()
    enabled = bool(settings.langsmith_api_key)
    if enabled:
        project = settings.langsmith_project
        return {
            "enabled": True,
            "project": project,
            "url": f"https://smith.langchain.com/projects/{project}",
        }
    return {"enabled": False, "project": settings.langsmith_project, "url": None}


def _reset_settings() -> None:
    """Reset cached settings. For use in tests only."""
    global _settings
    _settings = None
