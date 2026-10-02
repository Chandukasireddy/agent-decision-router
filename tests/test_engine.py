"""Tests for DecisionEngine routing pipeline and confidence gating."""

import pytest
from agent_decision_router.engine import DecisionEngine
from agent_decision_router.models import (
    AgentState,
    DecisionResult,
    DecisionStatus,
    RouteRequest,
    ToolCallRecord,
)
from agent_decision_router.registry import create_default_coding_registry
from agent_decision_router.cache import FastPathCache
from agent_decision_router.backend.mock import MockBackend


def test_engine_fast_path_cache():
    backend = MockBackend(canned_action="read_code_file", canned_confidence=0.88)
    cache = FastPathCache()
    engine = DecisionEngine(backend=backend, cache=cache)

    request = RouteRequest(state="Read the config.json file to inspect settings")

    # First call: Cache miss -> calls backend -> caches
    res1 = engine.route_sync(request)
    assert res1.action == "read_code_file"
    assert res1.status == DecisionStatus.EXECUTABLE
    assert res1.source == "system1"
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0

    # Second call: Cache hit -> returns cached result immediately
    res2 = engine.route_sync(request)
    assert res2.action == "read_code_file"
    assert res2.status == DecisionStatus.CACHE_HIT
    assert res2.source == "cache"
    assert cache.stats.hits == 1


def test_engine_high_confidence_executable():
    # Confidence >= 0.70
    backend = MockBackend(canned_action="run_terminal_command", canned_confidence=0.85)
    engine = DecisionEngine(backend=backend)

    res = engine.route_sync("Run pytest on unit tests")
    assert res.action == "run_terminal_command"
    assert res.status == DecisionStatus.EXECUTABLE
    assert res.confidence == 0.85


def test_engine_medium_confidence_triggers_lookahead():
    # 0.40 <= confidence < 0.70
    backend = MockBackend(
        canned_action="apply_code_patch",
        canned_confidence=0.55,
        canned_probabilities={"apply_code_patch": 0.55, "read_code_file": 0.45},
    )
    engine = DecisionEngine(backend=backend)

    res = engine.route_sync("Fix the broken authentication loop in server.py")
    assert res.status == DecisionStatus.LOOKAHEAD_RESOLVED
    assert res.source == "lookahead"
    assert res.confidence >= 0.70


def test_engine_low_confidence_declines():
    # Confidence < 0.40 -> Must decline execution
    backend = MockBackend(
        canned_action="run_terminal_command",
        canned_confidence=0.32,
        canned_probabilities={
            "run_terminal_command": 0.32,
            "read_code_file": 0.28,
            "apply_code_patch": 0.25,
            "search_codebase": 0.15,
        },
    )
    engine = DecisionEngine(backend=backend)

    res = engine.route_sync("Do something with the code repository")
    assert res.action == "DECLINE"
    assert res.status == DecisionStatus.DECLINED
    assert res.confidence == 0.32
    assert "Refusing weak match" in (res.reason or "")


def test_engine_destructive_requires_confirmation():
    backend = MockBackend(
        canned_action="run_terminal_command",
        canned_confidence=0.90,
        canned_destructive=True,
    )
    engine = DecisionEngine(backend=backend)

    res = engine.route_sync("Remove the temp directory and reset git HEAD")
    assert res.action == "run_terminal_command"
    assert res.is_destructive is True
    assert res.requires_confirmation is True


def test_engine_task_completion():
    backend = MockBackend(
        canned_action="run_terminal_command",
        canned_confidence=0.50,
        canned_completion=True,
    )
    engine = DecisionEngine(backend=backend)

    res = engine.route_sync("All tests have passed and changes are verified.")
    assert res.action == "COMPLETE_TASK"
    assert res.status == DecisionStatus.TASK_COMPLETED
    assert res.task_completion is True


def test_engine_pre_decision_and_post_tool_lifecycle():
    engine = DecisionEngine(backend=MockBackend(canned_action="read_code_file", canned_confidence=0.85))

    # Pre-decision hook that intercepts specific queries
    def interceptor(req: RouteRequest):
        if "intercept_me" in str(req.state):
            return DecisionResult(
                action="ask_user_clarification",
                confidence=1.0,
                status=DecisionStatus.EXECUTABLE,
                reason="Intercepted by rule",
            )
        return None

    engine.add_pre_decision_hook(interceptor)

    res = engine.route_sync("intercept_me please")
    assert res.action == "ask_user_clarification"
    assert res.reason == "Intercepted by rule"

    # Post-tool hook
    recorded_records = []
    engine.add_post_tool_hook(lambda rec, state: recorded_records.append(rec))

    state = AgentState(user_goal="Test lifecycle")
    engine.record_tool_result(
        tool="read_code_file",
        output="contents of file",
        arguments={"path": "main.py"},
        state=state,
    )

    assert len(recorded_records) == 1
    assert recorded_records[0].tool == "read_code_file"
    assert len(state.recent_tools) == 1
