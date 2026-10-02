"""Base interface and normalization for decision model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from agent_decision_router.models import DecisionPayload


class ParsedDecision(BaseModel):
    """Normalized response parsed from a System 1 decision model."""
    selected_action: str = Field(..., description="Action or tool selected by model")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the choice (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across candidate criteria"
    )
    is_destructive: bool = Field(
        default=False,
        description="Whether planned action was predicted as destructive"
    )
    destructive_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence on is_destructive"
    )
    task_completion: bool = Field(
        default=False,
        description="Whether user task is judged complete"
    )
    task_completion_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence on task_completion"
    )
    raw_response: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw backend response"
    )


class BaseDecisionBackend(ABC):
    """Abstract base class for System 1 decision backends."""

    @abstractmethod
    async def query_decision(self, payload: DecisionPayload) -> ParsedDecision:
        """Asynchronously query the decision model with state and typed questions."""
        pass

    @abstractmethod
    def query_decision_sync(self, payload: DecisionPayload) -> ParsedDecision:
        """Synchronously query the decision model with state and typed questions."""
        pass

    def parse_systemone_response(
        self,
        raw_result: Dict[str, Any],
        payload: DecisionPayload,
    ) -> ParsedDecision:
        """Normalize typical System 1 / Jev / clef-flash JSON responses."""
        # Result may be nested under 'result' or 'answers'
        data = raw_result
        if isinstance(data, dict):
            if "result" in data and isinstance(data["result"], dict):
                data = data["result"]
            if "answers" in data and isinstance(data["answers"], dict):
                data = data["answers"]

        # Parse next_tool / next_action choice
        next_tool_data = data.get("next_tool") or data.get("next_action") or {}
        selected_action = "unknown"
        probabilities: Dict[str, float] = {}
        confidence = 0.0

        if isinstance(next_tool_data, dict):
            # Format: {"choice": "tool_name", "probabilities": {"tool_name": 0.85, ...}}
            selected_action = (
                next_tool_data.get("choice")
                or next_tool_data.get("selected")
                or next_tool_data.get("answer")
                or ""
            )
            raw_probs = (
                next_tool_data.get("probabilities")
                or next_tool_data.get("probs")
                or next_tool_data.get("scores")
                or {}
            )
            if isinstance(raw_probs, dict):
                for k, v in raw_probs.items():
                    try:
                        probabilities[str(k)] = float(v)
                    except (ValueError, TypeError):
                        pass

            if "confidence" in next_tool_data:
                try:
                    confidence = float(next_tool_data["confidence"])
                except (ValueError, TypeError):
                    pass
        elif isinstance(next_tool_data, str):
            selected_action = next_tool_data

        # If confidence wasn't explicit, derive from probabilities
        if not confidence and probabilities and selected_action in probabilities:
            confidence = probabilities[selected_action]
        elif not confidence and probabilities:
            confidence = max(probabilities.values())
        elif not confidence and selected_action != "unknown":
            confidence = 0.75  # Default baseline if backend returned bare string

        # Ensure probabilities contains selected action
        if selected_action and selected_action not in probabilities:
            probabilities[selected_action] = confidence

        # Parse is_destructive (noul)
        is_destructive, dest_conf = self._parse_noul_field(data.get("is_destructive"))

        # Parse task_completion (noul)
        task_completion, comp_conf = self._parse_noul_field(data.get("task_completion"))

        # If selected action is empty or still unknown, pick highest prob key if available
        if (not selected_action or selected_action == "unknown") and probabilities:
            selected_action = max(probabilities, key=lambda k: probabilities[k])
            confidence = probabilities[selected_action]

        return ParsedDecision(
            selected_action=selected_action,
            confidence=max(0.0, min(1.0, confidence)),
            probabilities=probabilities,
            is_destructive=is_destructive,
            destructive_confidence=dest_conf,
            task_completion=task_completion,
            task_completion_confidence=comp_conf,
            raw_response=raw_result,
        )

    def _parse_noul_field(self, field_value: Any) -> tuple[bool, float]:
        """Extract boolean decision and confidence from a noul response."""
        if field_value is None:
            return False, 0.0

        if isinstance(field_value, bool):
            return field_value, 1.0 if field_value else 0.0

        if isinstance(field_value, (int, float)):
            conf = float(field_value)
            return conf >= 0.5, conf

        if isinstance(field_value, dict):
            # Format: {"type": "noul", "noul": 0.0061} or {"value": true, "prob": 0.9}
            val = (
                field_value.get("noul")
                if "noul" in field_value
                else field_value.get("value", field_value.get("answer"))
            )
            conf = field_value.get("confidence") or field_value.get("prob") or field_value.get("score")

            if isinstance(val, (int, float)):
                conf_float = max(0.0, min(1.0, float(val)))
                is_true = conf_float >= 0.5
                return is_true, conf_float

            is_true = False
            if isinstance(val, bool):
                is_true = val
            elif isinstance(val, str):
                is_true = val.lower() in ("true", "yes", "1", "positive")

            conf_float = 0.0
            if conf is not None:
                try:
                    conf_float = max(0.0, min(1.0, float(conf)))
                except (ValueError, TypeError):
                    conf_float = 1.0 if is_true else 0.0
            else:
                conf_float = 1.0 if is_true else 0.0

            return is_true, conf_float

        if isinstance(field_value, str):
            is_true = field_value.lower() in ("true", "yes", "1", "destructive", "completed")
            return is_true, 0.8 if is_true else 0.0

        return False, 0.0
