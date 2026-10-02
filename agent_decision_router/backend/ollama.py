"""Local Ollama / System One client for running local routing models (e.g., nimble)."""

from __future__ import annotations

from typing import Any, Dict, Optional
import httpx

from agent_decision_router.config import get_settings
from agent_decision_router.models import DecisionPayload
from agent_decision_router.backend.base import BaseDecisionBackend, ParsedDecision


class OllamaAPIError(Exception):
    """Exception raised when local Ollama System One API returns an error or fails."""
    pass


class OllamaBackend(BaseDecisionBackend):
    """Client for local Ollama /v1/systemone endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout

    @property
    def endpoint_url(self) -> str:
        """Construct the REST URL for local Ollama System One endpoint."""
        return f"{self.base_url}/v1/systemone"

    def _prepare_payload(self, payload: DecisionPayload) -> Dict[str, Any]:
        """Convert DecisionPayload to System One JSON schema."""
        questions_dict: Dict[str, Any] = {}
        for q_name, q_def in payload.questions.items():
            q_obj: Dict[str, Any] = {
                "type": q_def.type,
                "instructions": q_def.instructions,
            }
            if q_def.criteria:
                q_obj["criteria"] = q_def.criteria
            questions_dict[q_name] = q_obj

        return {
            "model": self.model,
            "state": payload.state,
            "questions": questions_dict,
        }

    async def query_decision(self, payload: DecisionPayload) -> ParsedDecision:
        """Asynchronously call local Ollama /v1/systemone endpoint."""
        url = self.endpoint_url
        body = self._prepare_payload(payload)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=body)
            except httpx.RequestError as exc:
                raise OllamaAPIError(
                    f"Failed to connect to local Ollama System One at {url}: {exc}"
                ) from exc

        return self._process_response(response, payload)

    def query_decision_sync(self, payload: DecisionPayload) -> ParsedDecision:
        """Synchronously call local Ollama /v1/systemone endpoint."""
        url = self.endpoint_url
        body = self._prepare_payload(payload)

        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = client.post(url, json=body)
            except httpx.RequestError as exc:
                raise OllamaAPIError(
                    f"Failed to connect to local Ollama System One at {url}: {exc}"
                ) from exc

        return self._process_response(response, payload)

    def _process_response(
        self,
        response: httpx.Response,
        payload: DecisionPayload,
    ) -> ParsedDecision:
        """Validate and parse HTTP response from Ollama System One."""
        if response.status_code != 200:
            raise OllamaAPIError(
                f"Ollama System One returned status {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise OllamaAPIError(f"Failed to parse Ollama JSON response: {exc}") from exc

        return self.parse_systemone_response(data, payload)
