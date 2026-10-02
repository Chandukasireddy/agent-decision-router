"""Stage 3 Fallback: Bounded shallow lookahead / MCTS heuristic evaluator."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent_decision_router.models import (
    AgentState,
    DecisionResult,
    DecisionStatus,
    ToolCallRecord,
)
from agent_decision_router.backend.base import ParsedDecision
from agent_decision_router.registry import ToolRegistry


class MCTSNode(BaseModel):
    """Search node for shallow Monte Carlo Tree Search lookahead."""
    action: str
    prior_prob: float
    visit_count: int = 0
    total_value: float = 0.0
    heuristic_score: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return self.heuristic_score
        return (self.total_value + self.heuristic_score) / (self.visit_count + 1)

    def ucb1_score(self, total_visits: int, exploration_weight: float = 1.414) -> float:
        if self.visit_count == 0:
            return float("inf")
        exploration = exploration_weight * math.sqrt(math.log(total_visits) / self.visit_count)
        return self.q_value + exploration


class LookaheadEvaluator:
    """Bounded shallow lookahead evaluator for medium-confidence decisions (0.40 <= conf < 0.70)."""

    def __init__(
        self,
        registry: ToolRegistry,
        max_depth: int = 2,
        simulations_per_candidate: int = 15,
        exploration_weight: float = 1.0,
    ) -> None:
        self.registry = registry
        self.max_depth = max_depth
        self.simulations = simulations_per_candidate
        self.exploration_weight = exploration_weight

    def evaluate(
        self,
        parsed: ParsedDecision,
        state: str | AgentState,
    ) -> DecisionResult:
        """Run bounded lookahead over top candidate actions to resolve ambiguity.

        Args:
            parsed: Parsed System 1 decision with medium confidence.
            state: Current environment state.

        Returns:
            Resolved DecisionResult with updated confidence and rationale.
        """
        # Extract candidate actions and probabilities
        probs = parsed.probabilities or {parsed.selected_action: parsed.confidence}
        sorted_candidates = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        # Select top-K candidates (at least top 2 or candidates with prob >= 0.20)
        top_candidates = [
            (action, prob) for action, prob in sorted_candidates[:3]
            if prob >= 0.15 or action == parsed.selected_action
        ]

        if not top_candidates:
            top_candidates = [(parsed.selected_action, parsed.confidence)]

        # If only one candidate exists, return directly
        if len(top_candidates) == 1:
            action, prob = top_candidates[0]
            boosted_conf = min(0.72, prob + 0.15)
            return DecisionResult(
                action=action,
                confidence=boosted_conf,
                status=DecisionStatus.LOOKAHEAD_RESOLVED,
                is_destructive=parsed.is_destructive,
                destructive_confidence=parsed.destructive_confidence,
                task_completion=parsed.task_completion,
                task_completion_confidence=parsed.task_completion_confidence,
                probabilities=probs,
                reason=f"Lookahead single dominant candidate resolved: {action}",
                source="lookahead",
            )

        # Initialize MCTS root candidate nodes
        nodes: Dict[str, MCTSNode] = {}
        for action, prob in top_candidates:
            h_score = self._compute_heuristic_value(action, prob, state)
            nodes[action] = MCTSNode(
                action=action,
                prior_prob=prob,
                heuristic_score=h_score,
            )

        # Run bounded simulation loops
        total_visits = 0
        total_sims = self.simulations * len(top_candidates)

        for _ in range(total_sims):
            # Select node with highest UCB1
            selected_action = max(
                nodes.keys(),
                key=lambda a: nodes[a].ucb1_score(max(1, total_visits), self.exploration_weight)
            )
            node = nodes[selected_action]

            # Simulate shallow rollout (1-2 transitions)
            sim_value = self._simulate_rollout(node.action, state, self.max_depth)

            # Backpropagate
            node.visit_count += 1
            node.total_value += sim_value
            total_visits += 1

        # Select best candidate by visits and Q-value
        best_action = max(
            nodes.keys(),
            key=lambda a: (nodes[a].visit_count, nodes[a].q_value)
        )
        best_node = nodes[best_action]

        # Calculate resolved confidence combining System 1 prior and lookahead value
        # Weight prior 40%, lookahead value 60%
        normalized_q = max(0.0, min(1.0, (best_node.q_value + 1.0) / 2.0))
        resolved_confidence = round((best_node.prior_prob * 0.40) + (normalized_q * 0.60), 3)

        # Ensure lookahead elevates above threshold if search was decisive
        if best_node.visit_count > total_visits * 0.50:
            resolved_confidence = max(resolved_confidence, 0.72)

        # Status: If resolved confidence is at least 0.70, it's LOOKAHEAD_RESOLVED
        status = (
            DecisionStatus.LOOKAHEAD_RESOLVED
            if resolved_confidence >= 0.70
            else DecisionStatus.DECLINED
        )

        reason = (
            f"Lookahead evaluated {len(top_candidates)} candidates across {total_visits} simulations. "
            f"Candidate '{best_action}' selected (visits: {best_node.visit_count}, Q-value: {round(best_node.q_value, 2)})."
        )

        return DecisionResult(
            action=best_action,
            confidence=resolved_confidence,
            status=status,
            is_destructive=parsed.is_destructive,
            destructive_confidence=parsed.destructive_confidence,
            task_completion=parsed.task_completion,
            task_completion_confidence=parsed.task_completion_confidence,
            probabilities=probs,
            reason=reason,
            source="lookahead",
            metadata={
                "mcts_nodes": {
                    k: {
                        "visits": v.visit_count,
                        "q_value": round(v.q_value, 3),
                        "prior": v.prior_prob,
                    }
                    for k, v in nodes.items()
                }
            }
        )

    def _compute_heuristic_value(
        self,
        action: str,
        prior_prob: float,
        state: str | AgentState,
    ) -> float:
        """Heuristic evaluation of action suitability given current trajectory context."""
        state_str = state.to_state_string() if isinstance(state, AgentState) else str(state)
        state_lower = state_str.lower()
        score = prior_prob * 1.5

        # Extract recent tool calls if AgentState
        recent_tools: List[ToolCallRecord] = []
        if isinstance(state, AgentState):
            recent_tools = state.recent_tools

        tool_names_recent = [t.tool for t in recent_tools[-3:]] if recent_tools else []

        # Heuristic Rule 1: Read before Patch / Write
        # If attempting to patch or edit a file, has a read tool been executed recently?
        has_read = any("read" in t or "inspect" in t for t in tool_names_recent)
        has_file_in_state = bool(re.search(r"[\w_-]+\.(py|js|ts|json|md|rs|go)\b", state_str))

        if "patch" in action or "apply" in action or "write" in action:
            if not has_read and has_file_in_state:
                # Penalize blind patching before reading
                score -= 0.6
            else:
                score += 0.4
        elif "read" in action or "inspect" in action:
            if not has_read and has_file_in_state:
                # Reward reading/inspecting the unread file
                score += 0.6

        # Heuristic Rule 2: Don't repeat the exact same failed action in an immediate loop
        if recent_tools and not recent_tools[-1].success:
            last_failed_tool = recent_tools[-1].tool
            if action == last_failed_tool:
                score -= 0.8  # Strong penalty for mindless repetition without diagnosis
            elif "read" in action or "inspect" in action or "search" in action:
                score += 0.6  # Diagnosis rewarded after failure

        # Heuristic Rule 3: If test output shows assertion failure, prioritize inspecting file or patching
        if "assertionerror" in state_lower or "failed:" in state_lower:
            if "read" in action or "patch" in action:
                score += 0.5
            elif "terminal" in action:
                # Running tests again without edits is usually useless
                score -= 0.3

        # Heuristic Rule 4: Destructive penalty in ambiguous state
        tool_def = self.registry.get_tool(action)
        if tool_def and tool_def.is_destructive:
            score -= 0.2

        return score

    def _simulate_rollout(
        self,
        action: str,
        state: str | AgentState,
        remaining_depth: int,
    ) -> float:
        """Simulate a short 1-2 step rollout evaluating future state alignment."""
        if remaining_depth <= 0:
            return 0.5

        state_str = state.to_state_string() if isinstance(state, AgentState) else str(state)
        recent_tools: List[ToolCallRecord] = state.recent_tools if isinstance(state, AgentState) else []
        tool_names_recent = [t.tool for t in recent_tools[-3:]]
        has_read = any("read" in t or "inspect" in t for t in tool_names_recent)

        value = 0.5

        # Positive progression indicators
        state_lower = state_str.lower()
        if "test" in state_lower and "run" in action:
            value += 0.3
        elif any(k in state_lower for k in ("fix", "update", "modify", "edit", "implement", "change", "refactor")):
            if ("patch" in action or "write" in action) and not has_read:
                value -= 0.3  # Penalize blind patch before reading
            elif "read" in action or "inspect" in action:
                value += 0.4
            elif "patch" in action:
                value += 0.2

        return min(1.0, max(-1.0, value))
