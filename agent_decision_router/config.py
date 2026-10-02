"""Configuration management for agent-decision-router."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from current working directory or package root if available
load_dotenv(dotenv_path=Path.cwd() / ".env")


class Settings:
    """Application settings with environment variable fallbacks."""

    def __init__(self) -> None:
        # Cloudflare Workers AI
        self.cloudflare_account_id: str = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.cloudflare_api_token: str = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        self.cloudflare_model: str = os.getenv("CLOUDFLARE_MODEL", "@cf/cloudflare/clef-flash").strip()

        # Local Ollama / System One
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "nimble").strip()

        # Router Backend: 'cloudflare', 'ollama', or 'mock'
        self.router_backend: str = os.getenv("ROUTER_BACKEND", "cloudflare").strip().lower()

        # Decision Confidence Thresholds
        self.confidence_threshold_high: float = float(os.getenv("CONFIDENCE_THRESHOLD_HIGH", "0.70"))
        self.confidence_threshold_low: float = float(os.getenv("CONFIDENCE_THRESHOLD_LOW", "0.40"))

        # Fast-Path Cache
        self.fast_path_cache_enabled: bool = os.getenv("FAST_PATH_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
        self.fast_path_cache_ttl_seconds: int = int(os.getenv("FAST_PATH_CACHE_TTL_SECONDS", "3600"))
        self.fast_path_cache_max_size: int = int(os.getenv("FAST_PATH_CACHE_MAX_SIZE", "1024"))

        # Server
        self.router_host: str = os.getenv("ROUTER_HOST", "0.0.0.0")
        self.router_port: int = int(os.getenv("ROUTER_PORT", "8000"))

    @property
    def is_cloudflare_configured(self) -> bool:
        """Check if Cloudflare credentials are provided."""
        return bool(self.cloudflare_account_id and self.cloudflare_api_token)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Retrieve singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)
    _settings = Settings()
    return _settings
