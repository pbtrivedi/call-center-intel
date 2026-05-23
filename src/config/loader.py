import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

_SETTINGS_FILE = Path(__file__).parent / "settings.yaml"
_settings: "Settings | None" = None


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
    langsmith_api_key: str
    openai_api_key: str
    gemini_api_key: str
    groq_api_key: str


def get_settings() -> Settings:
    global _settings
    if _settings is not None:
        return _settings

    load_dotenv()
    with open(_SETTINGS_FILE) as f:
        defaults = yaml.safe_load(f)

    _settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", defaults["llm_provider"]),
        whisper_model=os.getenv("WHISPER_MODEL", defaults["whisper_model"]),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", defaults["max_file_size_mb"])),
        max_duration_minutes=int(os.getenv("MAX_DURATION_MINUTES", defaults["max_duration_minutes"])),
        db_path=os.getenv("DB_PATH", defaults["db_path"]),
        app_port=int(os.getenv("APP_PORT", defaults["app_port"])),
        max_temp_files=int(os.getenv("MAX_TEMP_FILES", defaults["max_temp_files"])),
        log_level=os.getenv("LOG_LEVEL", defaults["log_level"]),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", defaults["langsmith_project"]),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
    )
    return _settings


def _reset_settings() -> None:
    """Reset cached settings. For use in tests only."""
    global _settings
    _settings = None
