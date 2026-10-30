"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root (two levels up from this file: app/core/config.py → app/ → backend/ → project)
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_PROJECT_ROOT = _BACKEND_DIR.parent  # project root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_ENV: str = "development"
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Security
    SECRET_KEY: str = "change-this-in-production-use-a-long-random-string"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Groq LLM
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-8b-8192"
    GROQ_TIMEOUT_SECONDS: int = 30
    GROQ_MAX_TOKENS: int = 1200

    # Model storage
    MODELS_DIR: str = "models_store"

    # Dataset paths — defaults resolved relative to project root
    SEPSIS_CSV: str = str(_PROJECT_ROOT / "synthetic_sepsis_training.csv")
    AKI_CSV: str = str(_PROJECT_ROOT / "synthetic_aki_training.csv")
    CARDIO_CSV: str = str(_PROJECT_ROOT / "synthetic_cardiovascular_risk_training.csv")
    LIVER_CSV: str = str(_PROJECT_ROOT / "synthetic_liver_risk_training.csv")
    NEURO_CSV: str = str(_PROJECT_ROOT / "synthetic_neurological_risk_training.csv")
    RESP_CSV: str = str(_PROJECT_ROOT / "synthetic_respiratory_risk_training.csv")
    ANTIBIOTIC_CSV: str = str(_PROJECT_ROOT / "synthetic_antibiotic.csv")

    @property
    def models_dir_path(self) -> Path:
        p = Path(self.MODELS_DIR)
        if not p.is_absolute():
            p = _BACKEND_DIR / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def groq_configured(self) -> bool:
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY != "your_groq_api_key_here")


@lru_cache
def get_settings() -> Settings:
    return Settings()
