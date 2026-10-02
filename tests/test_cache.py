"""Tests for Stage 1 FastPathCache."""

import time
import pytest
from agent_decision_router.cache import FastPathCache
from agent_decision_router.models import (
    AgentState,
    DecisionResult,
    DecisionStatus,
    ToolCallRecord,
)


def test_state_hash_determinism():
    state1 = "User requested: Run pytest on test_auth.py"
    state2 = "User requested: Run pytest on test_auth.py"
    state3 = "User requested: Run pytest on other_test.py"

    criteria = {"run_terminal_command": "Run tests", "read_code_file": "Read files"}

    hash1 = FastPathCache.compute_state_hash(state1, criteria)
    hash2 = FastPathCache.compute_state_hash(state2, criteria)
    hash3 = FastPathCache.compute_state_hash(state3, criteria)

    assert hash1 == hash2
    assert hash1 != hash3


def test_state_hash_structured_agent_state():
    state = AgentState(
        user_goal="Fix syntax error",
        recent_tools=[
            ToolCallRecord(tool="run_terminal_command", output="SyntaxError at line 42", success=False)
        ],
        environment={"git_status": "clean"},
    )
    criteria = {"read_code_file": "Read code"}

    h1 = FastPathCache.compute_state_hash(state, criteria)
    h2 = FastPathCache.compute_state_hash(state, criteria)
    assert h1 == h2

    # Change tool output in state -> hash should change
    state.recent_tools[0].output = "Resolved"
    h3 = FastPathCache.compute_state_hash(state, criteria)
    assert h1 != h3


def test_cache_hit_and_miss():
    cache = FastPathCache(max_size=10, ttl_seconds=60)
    decision = DecisionResult(
        action="read_code_file",
        confidence=0.92,
        status=DecisionStatus.EXECUTABLE,
    )

    state_hash = "abc123hash"
    assert cache.get(state_hash) is None
    assert cache.stats.misses == 1

    cache.set(state_hash, decision)
    cached = cache.get(state_hash)
    assert cached is not None
    assert cached.action == "read_code_file"
    assert cached.status == DecisionStatus.CACHE_HIT
    assert cached.source == "cache"
    assert cache.stats.hits == 1
    assert cache.stats.hit_rate == 0.5


def test_cache_ttl_expiration():
    cache = FastPathCache(max_size=10, ttl_seconds=1)
    decision = DecisionResult(
        action="run_terminal_command",
        confidence=0.88,
        status=DecisionStatus.EXECUTABLE,
    )

    cache.set("hash_ttl", decision)
    assert cache.get("hash_ttl") is not None

    time.sleep(1.1)
    assert cache.get("hash_ttl") is None


def test_cache_lru_eviction():
    cache = FastPathCache(max_size=2, ttl_seconds=60)
    d1 = DecisionResult(action="tool1", confidence=0.8, status=DecisionStatus.EXECUTABLE)
    d2 = DecisionResult(action="tool2", confidence=0.8, status=DecisionStatus.EXECUTABLE)
    d3 = DecisionResult(action="tool3", confidence=0.8, status=DecisionStatus.EXECUTABLE)

    cache.set("h1", d1)
    cache.set("h2", d2)
    # Access h1 to make h2 the LRU
    assert cache.get("h1") is not None

    # Adding h3 should evict h2
    cache.set("h3", d3)
    assert cache.get("h2") is None
    assert cache.get("h1") is not None
    assert cache.get("h3") is not None
    assert cache.stats.evictions == 1
