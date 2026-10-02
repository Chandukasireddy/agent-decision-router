"""Cloudflare Workers AI client for running @cf/cloudflare/clef-flash."""

from __future__ import annotations

from typing import Any, Dict, Optional
import httpx

from agent_decision_router.config import get_settings
from agent_decision_router.models import DecisionPayload
from agent_decision_router.backend.base import BaseDecisionBackend, ParsedDecision


class CloudflareAPIError(Exception):
    """Exception raised when Cloudflare Workers AI API returns an error or fails."""
    pass


class CloudflareBackend(BaseDecisionBackend):
    """Client for Cloudflare Workers AI REST API running @cf/cloudflare/clef-flash."""

    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        settings = get_settings()
        self.account_id = account_id or settings.cloudflare_account_id
        self.api_token = api_token or settings.cloudflare_api_token
        self.model = model or settings.cloudflare_model
        self.timeout = timeout

    @property
    def endpoint_url(self) -> str:
        """Construct the REST URL for Cloudflare Workers AI model run."""
        if not self.account_id:
            raise CloudflareAPIError(
                "CLOUDFLARE_ACCOUNT_ID is not configured. Set it in .env or pass to CloudflareBackend."
            )
        # Model path e.g. @cf/cloudflare/clef-flash
        model_name = self.model.lstrip("/")
        return f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{model_name}"

    def _get_headers(self) -> Dict[str, str]:
        """Generate authentication headers."""
        if not self.api_token:
            raise CloudflareAPIError(
                "CLOUDFLARE_API_TOKEN is not configured. Set it in .env or pass to CloudflareBackend."
            )
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _prepare_payload(self, payload: DecisionPayload) -> Dict[str, Any]:
        """Convert DecisionPayload to Cloudflare REST JSON schema."""
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
            "state": payload.state,
            "questions": questions_dict,
        }

    async def query_decision(self, payload: DecisionPayload) -> ParsedDecision:
        """Asynchronously call Cloudflare Workers AI REST API."""
        url = self.endpoint_url
        headers = self._get_headers()
        body = self._prepare_payload(payload)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=body)
            except httpx.RequestError as exc:
                raise CloudflareAPIError(f"HTTP request to Cloudflare failed: {exc}") from exc

        return self._process_response(response, payload)

    def query_decision_sync(self, payload: DecisionPayload) -> ParsedDecision:
        """Synchronously call Cloudflare Workers AI REST API."""
        url = self.endpoint_url
        headers = self._get_headers()
        body = self._prepare_payload(payload)

        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = client.post(url, headers=headers, json=body)
            except httpx.RequestError as exc:
                raise CloudflareAPIError(f"HTTP request to Cloudflare failed: {exc}") from exc

        return self._process_response(response, payload)

    def _process_response(
        self,
        response: httpx.Response,
        payload: DecisionPayload,
    ) -> ParsedDecision:
        """Validate and parse HTTP response from Cloudflare."""
        if response.status_code != 200:
            error_text = response.text
            raise CloudflareAPIError(
                f"Cloudflare Workers AI returned status {response.status_code}: {error_text}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise CloudflareAPIError(f"Failed to parse Cloudflare JSON response: {exc}") from exc

        if isinstance(data, dict) and not data.get("success", True):
            errors = data.get("errors", [])
            raise CloudflareAPIError(f"Cloudflare Workers AI execution failed: {errors}")

        return self.parse_systemone_response(data, payload)
