"""Decision backends for agent-decision-router."""

from __future__ import annotations

from typing import Any, Optional
from agent_decision_router.config import get_settings
from agent_decision_router.backend.base import BaseDecisionBackend, ParsedDecision
from agent_decision_router.backend.cloudflare import CloudflareBackend, CloudflareAPIError
from agent_decision_router.backend.ollama import OllamaBackend, OllamaAPIError
from agent_decision_router.backend.mock import MockBackend

__all__ = [
    "BaseDecisionBackend",
    "ParsedDecision",
    "CloudflareBackend",
    "CloudflareAPIError",
    "OllamaBackend",
    "OllamaAPIError",
    "MockBackend",
    "get_backend",
]


def get_backend(
    name: Optional[str] = None,
    **kwargs: Any,
) -> BaseDecisionBackend:
    """Factory to retrieve configured decision backend.

    Args:
        name: Backend type ('cloudflare', 'ollama', or 'mock').
              Defaults to ROUTER_BACKEND from environment settings.
        **kwargs: Optional constructor arguments passed to the backend.

    Returns:
        BaseDecisionBackend instance.
    """
    settings = get_settings()
    backend_name = (name or settings.router_backend).strip().lower()

    if backend_name in ("cloudflare", "auto"):
        account_id = kwargs.get("account_id") or settings.cloudflare_account_id
        api_token = kwargs.get("api_token") or settings.cloudflare_api_token
        if not account_id or not api_token:
            # If backend was 'auto' or not explicitly overridden with credentials, fallback to mock with warning
            if backend_name == "auto" or kwargs.get("fallback_mock", True):
                return MockBackend(auto_heuristic=True)
        return CloudflareBackend(**kwargs)
    elif backend_name in ("ollama", "local", "systemone"):
        return OllamaBackend(**kwargs)
    elif backend_name in ("mock", "heuristic", "test"):
        return MockBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend_name}'. Supported backends: 'cloudflare', 'ollama', 'mock'"
        )
