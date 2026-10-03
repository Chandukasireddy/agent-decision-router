"""Base interface and normalization for decision model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
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
    selected_model: Optional[str] = Field(
        default=None,
        description="Public AI model selected by decision router"
    )
    model_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence of model choice"
    )
    model_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across candidate models"
    )
    selected_skill: Optional[str] = Field(
        default=None,
        description="Agent skill domain selected by decision router"
    )
    skill_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence of skill choice"
    )
    skill_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across candidate skills"
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

    def _parse_choice_field(self, field_data: Any) -> Tuple[str, float, Dict[str, float]]:
        """Parse a Choice question response containing choice, confidence, and probabilities."""
        if not field_data:
            return "", 0.0, {}

        selected = ""
        confidence = 0.0
        probabilities: Dict[str, float] = {}

        if isinstance(field_data, dict):
            selected = (
                field_data.get("choice")
                or field_data.get("selected")
                or field_data.get("answer")
                or ""
            )
            raw_probs = (
                field_data.get("probabilities")
                or field_data.get("probs")
                or field_data.get("scores")
                or {}
            )
            if isinstance(raw_probs, dict):
                for k, v in raw_probs.items():
                    try:
                        probabilities[str(k)] = float(v)
                    except (ValueError, TypeError):
                        pass

            if "confidence" in field_data:
                try:
                    confidence = float(field_data["confidence"])
                except (ValueError, TypeError):
                    pass
        elif isinstance(field_data, str):
            selected = field_data

        if not confidence and probabilities and selected in probabilities:
            confidence = probabilities[selected]
        elif not confidence and probabilities:
            confidence = max(probabilities.values())
        elif not confidence and selected:
            confidence = 0.75

        if selected and selected not in probabilities:
            probabilities[selected] = confidence

        if not selected and probabilities:
            selected = max(probabilities, key=lambda k: probabilities[k])
            confidence = probabilities[selected]

        return selected, max(0.0, min(1.0, confidence)), probabilities

    def parse_systemone_response(
        self,
        raw_result: Dict[str, Any],
        payload: DecisionPayload,
    ) -> ParsedDecision:
        """Normalize typical System 1 / Jev / clef-flash JSON responses."""
        data = raw_result
        if isinstance(data, dict):
            if "result" in data and isinstance(data["result"], dict):
                data = data["result"]
            if "answers" in data and isinstance(data["answers"], dict):
                data = data["answers"]

        action, action_conf, action_probs = self._parse_choice_field(
            data.get("next_tool") or data.get("next_action")
        )

        model_choice, model_conf, model_probs = self._parse_choice_field(
            data.get("model_tier")
        )

        skill_choice, skill_conf, skill_probs = self._parse_choice_field(
            data.get("selected_skill")
        )

        is_destructive, dest_conf = self._parse_noul_field(data.get("is_destructive"))
        task_completion, comp_conf = self._parse_noul_field(data.get("task_completion"))

        return ParsedDecision(
            selected_action=action or "unknown",
            confidence=action_conf,
            probabilities=action_probs,
            selected_model=model_choice or None,
            model_confidence=model_conf if model_choice else None,
            model_probabilities=model_probs,
            selected_skill=skill_choice or None,
            skill_confidence=skill_conf if skill_choice else None,
            skill_probabilities=skill_probs,
            is_destructive=is_destructive,
            destructive_confidence=dest_conf,
            task_completion=task_completion,
            task_completion_confidence=comp_conf,
            raw_response=raw_result,
        )

    def _parse_noul_field(self, field_value: Any) -> Tuple[bool, float]:
        """Extract boolean decision and confidence from a noul response."""
        if field_value is None:
            return False, 0.0

        if isinstance(field_value, bool):
            return field_value, 1.0 if field_value else 0.0

        if isinstance(field_value, (int, float)):
            conf = float(field_value)
            return conf >= 0.5, conf

        if isinstance(field_value, dict):
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
