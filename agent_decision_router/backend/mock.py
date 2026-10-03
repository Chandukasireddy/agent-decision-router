"""Mock and heuristic decision backend for deterministic testing and offline local development."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from agent_decision_router.models import DecisionPayload
from agent_decision_router.backend.base import BaseDecisionBackend, ParsedDecision


class MockBackend(BaseDecisionBackend):
    """Deterministic or heuristic mock backend for testing and development."""

    def __init__(
        self,
        canned_action: Optional[str] = None,
        canned_confidence: Optional[float] = None,
        canned_probabilities: Optional[Dict[str, float]] = None,
        canned_destructive: Optional[bool] = None,
        canned_completion: Optional[bool] = None,
        auto_heuristic: bool = True,
    ) -> None:
        self.canned_action = canned_action
        self.canned_confidence = canned_confidence
        self.canned_probabilities = canned_probabilities
        self.canned_destructive = canned_destructive
        self.canned_completion = canned_completion
        self.auto_heuristic = auto_heuristic

    def _generate_decision(self, payload: DecisionPayload) -> ParsedDecision:
        """Compute mock decision or run lightweight heuristic matching against state."""
        state_lower = payload.state.lower()
        next_tool_q = payload.questions.get("next_tool")
        criteria = next_tool_q.criteria if next_tool_q else {}
        candidate_keys = list(criteria.keys()) if criteria else []

        if not candidate_keys:
            candidate_keys = ["run_terminal_command", "read_code_file", "apply_code_patch"]

        # Check explicit overrides
        if self.canned_action:
            selected_action = self.canned_action
            conf = self.canned_confidence if self.canned_confidence is not None else 0.85
            probs = self.canned_probabilities or {
                selected_action: conf,
                **{k: round((1.0 - conf) / max(1, len(candidate_keys) - 1), 3) for k in candidate_keys if k != selected_action}
            }
            return ParsedDecision(
                selected_action=selected_action,
                confidence=conf,
                probabilities=probs,
                is_destructive=self.canned_destructive if self.canned_destructive is not None else False,
                destructive_confidence=0.9 if self.canned_destructive else 0.1,
                task_completion=self.canned_completion if self.canned_completion is not None else False,
                task_completion_confidence=0.95 if self.canned_completion else 0.05,
                raw_response={"mock": True},
            )

        # Heuristic detection for task completion
        completion_patterns = [
            r"all tests passed",
            r"fixed and verified",
            r"task is complete",
            r"objective satisfied",
            r"done",
        ]
        is_task_complete = any(re.search(pat, state_lower) for pat in completion_patterns)
        comp_conf = 0.92 if is_task_complete else 0.05

        # Heuristic detection for destructive operations
        destructive_patterns = [
            r"\brm\b",
            r"\bdelete\b",
            r"\bdrop\b",
            r"\bgit reset\b",
            r"\boverwrite\b",
            r"\bdestroy\b",
            r"\bkill\b",
        ]
        is_destructive = any(re.search(pat, state_lower) for pat in destructive_patterns)
        dest_conf = 0.88 if is_destructive else 0.08

        # Heuristic scoring of candidate tools based on keyword overlap
        scores: Dict[str, float] = {}
        for tool_name, desc in (criteria.items() if criteria else []):
            score = 1.0  # baseline
            desc_words = set(re.findall(r"\w+", (desc + " " + tool_name).lower()))
            state_words = set(re.findall(r"\w+", state_lower))
            overlap = len(desc_words.intersection(state_words))
            score += overlap * 2.0

            # Boost specific tools on specific intent patterns
            if "test" in state_lower or "pytest" in state_lower or "run" in state_lower:
                if "terminal" in tool_name or "command" in tool_name or "bash" in tool_name:
                    score += 5.0
            if "inspect" in state_lower or "read" in state_lower or "view" in state_lower or ".py" in state_lower:
                if "read" in tool_name or "file" in tool_name:
                    score += 4.0
            if "fix" in state_lower or "patch" in state_lower or "modify" in state_lower or "write" in state_lower:
                if "patch" in tool_name or "write" in tool_name:
                    score += 4.5
            if "clarif" in state_lower or "ambiguous" in state_lower or "question" in state_lower or "unsure" in state_lower:
                if "clarif" in tool_name or "user" in tool_name:
                    score += 6.0

            scores[tool_name] = score

        if not scores:
            scores = {k: 1.0 for k in candidate_keys}

        # Normalize to probability distribution via temperature-scaled softmax
        import math
        temperature = 2.0
        max_score = max(scores.values())
        exp_scores = {k: math.exp((v - max_score) / temperature) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        probabilities = {k: round(v / total_exp, 4) for k, v in exp_scores.items()}

        # ----------------------------------------------------------------------
        # Model Selection Heuristics (Public AI Models)
        # ----------------------------------------------------------------------
        model_q = payload.questions.get("model_tier")
        selected_model: Optional[str] = None
        model_conf: Optional[float] = None
        model_probs: Dict[str, float] = {}
        if model_q and model_q.criteria:
            model_scores: Dict[str, float] = {k: 1.0 for k in model_q.criteria.keys()}
            if any(k in state_lower for k in ("reason", "math", "algorithm", "race condition", "deadlock", "proof", "logic")):
                model_scores["deepseek_r1"] = model_scores.get("deepseek_r1", 1.0) + 9.0
            elif any(k in state_lower for k in ("refactor", "architect", "security", "audit", "complex", "overhaul", "migration")):
                model_scores["claude_3_5_sonnet"] = model_scores.get("claude_3_5_sonnet", 1.0) + 8.5
            elif any(k in state_lower for k in ("screenshot", "image", "diagram", "ui", "mockup", "visual", "layout")):
                model_scores["gpt_4o"] = model_scores.get("gpt_4o", 1.0) + 8.0
            elif any(k in state_lower for k in ("typo", "greeting", "hello", "format", "simple", "hi", "docstring")):
                model_scores["claude_3_5_haiku"] = model_scores.get("claude_3_5_haiku", 1.0) + 8.0
            elif any(k in state_lower for k in ("test", "pytest", "fast", "search", "grep", "iterate", "run")):
                model_scores["gemini_2_0_flash"] = model_scores.get("gemini_2_0_flash", 1.0) + 7.5
            else:
                model_scores["claude_3_5_sonnet"] = model_scores.get("claude_3_5_sonnet", 1.0) + 5.0

            m_max = max(model_scores.values())
            m_exp = {k: math.exp((v - m_max) / 2.0) for k, v in model_scores.items()}
            m_total = sum(m_exp.values())
            model_probs = {k: round(v / m_total, 4) for k, v in m_exp.items()}
            selected_model = max(model_probs, key=lambda k: model_probs[k])
            model_conf = model_probs[selected_model]

        # ----------------------------------------------------------------------
        # Skill Selection Heuristics (Public Domain Skills)
        # ----------------------------------------------------------------------
        skill_q = payload.questions.get("selected_skill")
        selected_skill: Optional[str] = None
        skill_conf: Optional[float] = None
        skill_probs: Dict[str, float] = {}
        if skill_q and skill_q.criteria:
            skill_scores: Dict[str, float] = {k: 1.0 for k in skill_q.criteria.keys()}
            if "pytest" in state_lower or ("test" in state_lower and "audit" not in state_lower and "vulnerability" not in state_lower):
                skill_scores["test_and_verification"] = skill_scores.get("test_and_verification", 1.0) + 9.0
            elif "security" in state_lower or "auth" in state_lower or "token" in state_lower or "timing" in state_lower:
                skill_scores["security_and_auth_audit"] = skill_scores.get("security_and_auth_audit", 1.0) + 8.0
            elif "search" in state_lower or "find" in state_lower or "grep" in state_lower or "where" in state_lower:
                skill_scores["codebase_navigation"] = skill_scores.get("codebase_navigation", 1.0) + 7.0
            elif "refactor" in state_lower or "patch" in state_lower or "write" in state_lower:
                skill_scores["architectural_refactoring"] = skill_scores.get("architectural_refactoring", 1.0) + 7.0
            elif "terminal" in state_lower or "command" in state_lower or "docker" in state_lower or "install" in state_lower:
                skill_scores["system_and_devops"] = skill_scores.get("system_and_devops", 1.0) + 7.0
            elif "clarif" in state_lower or "ambiguous" in state_lower or "question" in state_lower or "confirm" in state_lower:
                skill_scores["user_consultation"] = skill_scores.get("user_consultation", 1.0) + 8.0

            s_max = max(skill_scores.values())
            s_exp = {k: math.exp((v - s_max) / 2.0) for k, v in skill_scores.items()}
            s_total = sum(s_exp.values())
            skill_probs = {k: round(v / s_total, 4) for k, v in s_exp.items()}
            selected_skill = max(skill_probs, key=lambda k: skill_probs[k])
            skill_conf = skill_probs[selected_skill]

        selected_action = max(probabilities, key=lambda k: probabilities[k])
        confidence = probabilities[selected_action]

        return ParsedDecision(
            selected_action=selected_action,
            confidence=confidence,
            probabilities=probabilities,
            selected_model=selected_model,
            model_confidence=model_conf,
            model_probabilities=model_probs,
            selected_skill=selected_skill,
            skill_confidence=skill_conf,
            skill_probabilities=skill_probs,
            is_destructive=is_destructive,
            destructive_confidence=dest_conf,
            task_completion=is_task_complete,
            task_completion_confidence=comp_conf,
            raw_response={"mock": True, "scores": scores},
        )

    async def query_decision(self, payload: DecisionPayload) -> ParsedDecision:
        return self._generate_decision(payload)

    def query_decision_sync(self, payload: DecisionPayload) -> ParsedDecision:
        return self._generate_decision(payload)
