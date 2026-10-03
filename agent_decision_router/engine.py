"""Core Routing Pipeline: 3-stage decision engine with confidence gating and lifecycle hooks."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Union
from agent_decision_router.config import Settings, get_settings
from agent_decision_router.models import (
    AgentState,
    DecisionPayload,
    DecisionResult,
    DecisionStatus,
    RouteRequest,
    ToolCallRecord,
)
from agent_decision_router.registry import ToolRegistry, create_default_coding_registry
from agent_decision_router.cache import FastPathCache
from agent_decision_router.lookahead import LookaheadEvaluator
from agent_decision_router.backend.base import BaseDecisionBackend, ParsedDecision
from agent_decision_router.backend import get_backend


PreDecisionHook = Callable[[RouteRequest], Optional[DecisionResult]]
PostToolHook = Callable[[ToolCallRecord, AgentState], None]


class DecisionEngine:
    """The central routing engine coordinating Fast-Path Cache, System 1 backend, and Confidence Gating."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        backend: Optional[BaseDecisionBackend] = None,
        cache: Optional[FastPathCache] = None,
        lookahead: Optional[LookaheadEvaluator] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or create_default_coding_registry()
        self.backend = backend or get_backend()
        self.cache = cache or FastPathCache(
            max_size=self.settings.fast_path_cache_max_size,
            ttl_seconds=self.settings.fast_path_cache_ttl_seconds,
            enabled=self.settings.fast_path_cache_enabled,
        )
        self.lookahead = lookahead or LookaheadEvaluator(registry=self.registry)

        self._pre_decision_hooks: List[PreDecisionHook] = []
        self._post_tool_hooks: List[PostToolHook] = []

    def add_pre_decision_hook(self, hook: PreDecisionHook) -> None:
        """Register a hook executed before System 1 routing. Can return early DecisionResult."""
        self._pre_decision_hooks.append(hook)

    def add_post_tool_hook(self, hook: PostToolHook) -> None:
        """Register a hook executed after tool execution to record trajectory and update cache."""
        self._post_tool_hooks.append(hook)

    def _normalize_request(
        self,
        request: Union[RouteRequest, str, AgentState],
    ) -> RouteRequest:
        """Coerce raw input into standard RouteRequest."""
        if isinstance(request, RouteRequest):
            return request
        if isinstance(request, AgentState):
            return RouteRequest(state=request)
        return RouteRequest(state=str(request))

    def _build_decision_payload(
        self,
        request: RouteRequest,
    ) -> tuple[DecisionPayload, str, Dict[str, str]]:
        """Construct the typed questions payload for the backend."""
        state_str = (
            request.state.to_state_string()
            if isinstance(request.state, AgentState)
            else str(request.state)
        )

        criteria_schema = self.registry.generate_criteria_schema(
            candidate_names=request.candidate_tools,
            include_skills=True,
        )

        questions = self.registry.generate_questions(
            candidate_names=request.candidate_tools,
            include_skills=True,
            include_models=True,
            include_skills_question=True,
        )

        payload = DecisionPayload(
            state=state_str,
            questions=questions,
        )

        return payload, state_str, criteria_schema

    def _build_execution_directive(
        self,
        action: str,
        selected_model: Optional[str],
        selected_skill: Optional[str],
        is_destructive: bool,
        requires_confirmation: bool,
    ) -> Dict[str, Any]:
        """Generate downstream execution directive for agent runtime."""
        return {
            "dispatch_model": selected_model or "gemini_2_0_flash",
            "active_skill": selected_skill or "general",
            "target_action": action,
            "sandbox_policy": "APPROVAL_REQUIRED" if requires_confirmation else ("WRITE" if is_destructive else "READ_ONLY"),
            "autonomous_execution": not requires_confirmation and action not in ("DECLINE", "COMPLETE_TASK"),
        }

    def _evaluate_confidence_gate(
        self,
        parsed: ParsedDecision,
        state: Union[str, AgentState],
        start_time: float,
        threshold_high: float,
        threshold_low: float,
    ) -> DecisionResult:
        """Stage 3: Confidence Gate routing logic."""
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # Destructive safety check
        tool_def = self.registry.get_tool(parsed.selected_action)
        is_destructive = parsed.is_destructive or (tool_def.is_destructive if tool_def else False)
        requires_confirmation = is_destructive and (parsed.destructive_confidence >= 0.5 or (tool_def and tool_def.is_destructive))

        directive = self._build_execution_directive(
            action=parsed.selected_action,
            selected_model=parsed.selected_model,
            selected_skill=parsed.selected_skill,
            is_destructive=is_destructive,
            requires_confirmation=requires_confirmation,
        )

        # Check 1: Terminal Task Completion
        if (
            parsed.task_completion
            and parsed.task_completion_confidence >= threshold_high
            and (parsed.confidence < threshold_high or parsed.task_completion_confidence > parsed.confidence)
        ):
            complete_directive = self._build_execution_directive(
                action="COMPLETE_TASK",
                selected_model=parsed.selected_model,
                selected_skill=parsed.selected_skill,
                is_destructive=False,
                requires_confirmation=False,
            )
            return DecisionResult(
                action="COMPLETE_TASK",
                confidence=parsed.task_completion_confidence,
                status=DecisionStatus.TASK_COMPLETED,
                is_destructive=False,
                task_completion=True,
                task_completion_confidence=parsed.task_completion_confidence,
                probabilities=parsed.probabilities,
                selected_model=parsed.selected_model,
                model_confidence=parsed.model_confidence,
                model_probabilities=parsed.model_probabilities,
                selected_skill=parsed.selected_skill,
                skill_confidence=parsed.skill_confidence,
                skill_probabilities=parsed.skill_probabilities,
                execution_directive=complete_directive,
                reason="Task objectives fully satisfied. Stopping tool execution.",
                source="system1",
                execution_time_ms=elapsed_ms,
                requires_confirmation=False,
            )

        # Check 2: High Confidence (>= threshold_high, default 0.70)
        if parsed.confidence >= threshold_high:
            return DecisionResult(
                action=parsed.selected_action,
                confidence=parsed.confidence,
                status=DecisionStatus.EXECUTABLE,
                is_destructive=is_destructive,
                destructive_confidence=parsed.destructive_confidence,
                task_completion=parsed.task_completion,
                task_completion_confidence=parsed.task_completion_confidence,
                probabilities=parsed.probabilities,
                selected_model=parsed.selected_model,
                model_confidence=parsed.model_confidence,
                model_probabilities=parsed.model_probabilities,
                selected_skill=parsed.selected_skill,
                skill_confidence=parsed.skill_confidence,
                skill_probabilities=parsed.skill_probabilities,
                execution_directive=directive,
                reason=f"High confidence ({parsed.confidence:.2f}) tool match.",
                source="system1",
                execution_time_ms=elapsed_ms,
                requires_confirmation=requires_confirmation,
            )

        # Check 3: Medium Confidence (threshold_low <= confidence < threshold_high)
        if parsed.confidence >= threshold_low:
            lookahead_result = self.lookahead.evaluate(parsed=parsed, state=state)
            lookahead_result.execution_time_ms = round((time.time() - start_time) * 1000, 2)
            lookahead_result.requires_confirmation = requires_confirmation
            lookahead_result.selected_model = parsed.selected_model
            lookahead_result.model_confidence = parsed.model_confidence
            lookahead_result.model_probabilities = parsed.model_probabilities
            lookahead_result.selected_skill = parsed.selected_skill
            lookahead_result.skill_confidence = parsed.skill_confidence
            lookahead_result.skill_probabilities = parsed.skill_probabilities
            lookahead_result.execution_directive = self._build_execution_directive(
                action=lookahead_result.action,
                selected_model=parsed.selected_model,
                selected_skill=parsed.selected_skill,
                is_destructive=lookahead_result.is_destructive,
                requires_confirmation=requires_confirmation,
            )
            return lookahead_result

        # Check 4: Low Confidence (< threshold_low, default 0.40)
        decline_directive = self._build_execution_directive(
            action="DECLINE",
            selected_model=parsed.selected_model,
            selected_skill=parsed.selected_skill,
            is_destructive=is_destructive,
            requires_confirmation=False,
        )
        return DecisionResult(
            action="DECLINE",
            confidence=parsed.confidence,
            status=DecisionStatus.DECLINED,
            is_destructive=is_destructive,
            destructive_confidence=parsed.destructive_confidence,
            task_completion=parsed.task_completion,
            task_completion_confidence=parsed.task_completion_confidence,
            probabilities=parsed.probabilities,
            selected_model=parsed.selected_model,
            model_confidence=parsed.model_confidence,
            model_probabilities=parsed.model_probabilities,
            selected_skill=parsed.selected_skill,
            skill_confidence=parsed.skill_confidence,
            skill_probabilities=parsed.skill_probabilities,
            execution_directive=decline_directive,
            reason=f"Refusing weak match: confidence ({parsed.confidence:.2f}) is below threshold ({threshold_low:.2f}) to prevent hallucinated tool calls.",
            source="system1",
            execution_time_ms=elapsed_ms,
            requires_confirmation=False,
            metadata={
                "candidate_count": len(parsed.probabilities),
                "top_candidate": parsed.selected_action,
            }
        )

    async def route(
        self,
        request: Union[RouteRequest, str, AgentState],
    ) -> DecisionResult:
        """Asynchronous routing pipeline through Stages 1, 2, and 3."""
        start_time = time.time()
        req = self._normalize_request(request)

        # Run PreDecision hooks
        for hook in self._pre_decision_hooks:
            hook_result = hook(req)
            if hook_result is not None:
                hook_result.execution_time_ms = round((time.time() - start_time) * 1000, 2)
                return hook_result

        # Build payload and compute canonical state hash
        payload, state_str, criteria_schema = self._build_decision_payload(req)
        state_hash = FastPathCache.compute_state_hash(req.state, criteria_schema)

        # Stage 1: Fast-Path Cache
        if not req.skip_cache:
            cached_result = self.cache.get(state_hash)
            if cached_result is not None:
                return cached_result

        # Stage 2: System 1 Decision Model
        parsed = await self.backend.query_decision(payload)

        # Stage 3: Confidence Gate
        threshold_high = req.confidence_threshold_high or self.settings.confidence_threshold_high
        threshold_low = req.confidence_threshold_low or self.settings.confidence_threshold_low

        decision = self._evaluate_confidence_gate(
            parsed=parsed,
            state=req.state,
            start_time=start_time,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
        )

        # Populate cache on viable decisions
        if not req.skip_cache and decision.status in (
            DecisionStatus.EXECUTABLE,
            DecisionStatus.LOOKAHEAD_RESOLVED,
            DecisionStatus.TASK_COMPLETED,
        ):
            self.cache.set(state_hash, decision)

        return decision

    def route_sync(
        self,
        request: Union[RouteRequest, str, AgentState],
    ) -> DecisionResult:
        """Synchronous routing pipeline through Stages 1, 2, and 3."""
        start_time = time.time()
        req = self._normalize_request(request)

        # Run PreDecision hooks
        for hook in self._pre_decision_hooks:
            hook_result = hook(req)
            if hook_result is not None:
                hook_result.execution_time_ms = round((time.time() - start_time) * 1000, 2)
                return hook_result

        # Build payload and compute canonical state hash
        payload, state_str, criteria_schema = self._build_decision_payload(req)
        state_hash = FastPathCache.compute_state_hash(req.state, criteria_schema)

        # Stage 1: Fast-Path Cache
        if not req.skip_cache:
            cached_result = self.cache.get(state_hash)
            if cached_result is not None:
                return cached_result

        # Stage 2: System 1 Decision Model
        parsed = self.backend.query_decision_sync(payload)

        # Stage 3: Confidence Gate
        threshold_high = req.confidence_threshold_high or self.settings.confidence_threshold_high
        threshold_low = req.confidence_threshold_low or self.settings.confidence_threshold_low

        decision = self._evaluate_confidence_gate(
            parsed=parsed,
            state=req.state,
            start_time=start_time,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
        )

        # Populate cache on viable decisions
        if not req.skip_cache and decision.status in (
            DecisionStatus.EXECUTABLE,
            DecisionStatus.LOOKAHEAD_RESOLVED,
            DecisionStatus.TASK_COMPLETED,
        ):
            self.cache.set(state_hash, decision)

        return decision

    def record_tool_result(
        self,
        tool: str,
        output: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
        state: Optional[AgentState] = None,
    ) -> ToolCallRecord:
        """PostTool lifecycle hook: Record executed tool output into agent trajectory."""
        record = ToolCallRecord(
            tool=tool,
            arguments=arguments,
            output=output,
            success=success,
            error=error,
        )

        if state is not None:
            state.recent_tools.append(record)
            for hook in self._post_tool_hooks:
                hook(record, state)

        return record
