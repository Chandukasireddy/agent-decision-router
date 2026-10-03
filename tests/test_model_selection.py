"""Unit tests for Public AI Model Auto-Selector and Skill Routing."""

import pytest
from agent_decision_router.models import RouteRequest, DecisionStatus
from agent_decision_router.registry import create_default_coding_registry
from agent_decision_router.engine import DecisionEngine
from agent_decision_router.backend.mock import MockBackend


def test_model_auto_selection_simple_lookup():
    """Verify that simple typo / lookup tasks select claude_3_5_haiku or gemini_2_0_flash."""
    registry = create_default_coding_registry()
    backend = MockBackend()
    engine = DecisionEngine(registry=registry, backend=backend)

    res = engine.route_sync(RouteRequest(state="Fix minor typo in README.md header", skip_cache=True))
    assert res.selected_model in ("claude_3_5_haiku", "gemini_2_0_flash")
    assert res.execution_directive["dispatch_model"] in ("claude_3_5_haiku", "gemini_2_0_flash")


def test_model_auto_selection_complex_refactoring():
    """Verify that complex architectural refactoring prompts select claude_3_5_sonnet or deepseek_r1."""
    registry = create_default_coding_registry()
    backend = MockBackend()
    engine = DecisionEngine(registry=registry, backend=backend)

    prompt = "Refactor entire authentication architecture to zero-trust OAuth2 with PKCE, add security audit against timing attacks"
    res = engine.route_sync(RouteRequest(state=prompt, skip_cache=True))

    assert res.selected_model in ("claude_3_5_sonnet", "deepseek_r1")
    assert res.selected_skill in ("architectural_refactoring", "security_and_auth_audit")
    assert res.execution_directive["dispatch_model"] in ("claude_3_5_sonnet", "deepseek_r1")


def test_skill_selection_test_runner():
    """Verify that test runner commands activate test_and_verification skill."""
    registry = create_default_coding_registry()
    backend = MockBackend()
    engine = DecisionEngine(registry=registry, backend=backend)

    prompt = "Run pytest tests/test_auth.py to verify login fix"
    res = engine.route_sync(RouteRequest(state=prompt, skip_cache=True))

    assert res.selected_skill == "test_and_verification"
    assert res.action == "run_terminal_command"
    assert res.execution_directive["active_skill"] == "test_and_verification"
