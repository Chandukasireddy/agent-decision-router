"""Tests for Stage 3 Lookahead / MCTS heuristic evaluator."""

import pytest
from agent_decision_router.models import (
    AgentState,
    DecisionStatus,
    ToolCallRecord,
)
from agent_decision_router.registry import create_default_coding_registry
from agent_decision_router.backend.base import ParsedDecision
from agent_decision_router.lookahead import LookaheadEvaluator


def test_lookahead_evaluator_resolves_ambiguity():
    registry = create_default_coding_registry()
    evaluator = LookaheadEvaluator(registry=registry, simulations_per_candidate=10)

    # Medium confidence with 2 close candidate choices
    parsed = ParsedDecision(
        selected_action="apply_code_patch",
        confidence=0.52,
        probabilities={
            "apply_code_patch": 0.52,
            "read_code_file": 0.44,
            "ask_user_clarification": 0.04,
        },
        is_destructive=False,
    )

    state = AgentState(
        user_goal="Fix the assertion error in tests/auth_test.py",
        recent_tools=[
            ToolCallRecord(tool="run_terminal_command", output="AssertionError: 401 != 200", success=False)
        ],
    )

    result = evaluator.evaluate(parsed=parsed, state=state)
    assert result.status == DecisionStatus.LOOKAHEAD_RESOLVED
    assert result.confidence >= 0.70
    assert result.source == "lookahead"
    assert "Lookahead evaluated" in (result.reason or "")
    assert "mcts_nodes" in result.metadata


def test_lookahead_read_before_patch_heuristic():
    registry = create_default_coding_registry()
    evaluator = LookaheadEvaluator(registry=registry, simulations_per_candidate=20)

    # Model was slightly leaning to patch without reading first
    parsed = ParsedDecision(
        selected_action="apply_code_patch",
        confidence=0.51,
        probabilities={
            "apply_code_patch": 0.51,
            "read_code_file": 0.49,
        },
    )

    # No file has been read yet
    state = AgentState(
        user_goal="Update logic in auth_handler.py",
        recent_tools=[],
    )

    result = evaluator.evaluate(parsed=parsed, state=state)
    # Heuristic penalizes blind patch and favors reading first
    assert result.action == "read_code_file"
    assert result.status == DecisionStatus.LOOKAHEAD_RESOLVED
