"""Stage 1 Fast-Path Cache: Exact hash matching on state and recent tool outputs."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union
from agent_decision_router.models import AgentState, DecisionResult, DecisionStatus


class CacheStats:
    """Telemetry and metrics for the fast-path cache."""

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.total_latency_saved_ms: float = 0.0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.hits / self.total_requests, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
            "total_latency_saved_ms": round(self.total_latency_saved_ms, 2),
        }


class FastPathCache:
    """Content-addressable LRU cache with TTL for zero-latency instant routing."""

    def __init__(
        self,
        max_size: int = 1024,
        ttl_seconds: int = 3600,
        enabled: bool = True,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.stats = CacheStats()

    @staticmethod
    def compute_state_hash(
        state: Union[str, AgentState],
        criteria_schema: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate a deterministic canonical SHA-256 hash for the given state and criteria.

        Args:
            state: Plain string or structured AgentState.
            criteria_schema: Current active tool criteria mapping.

        Returns:
            64-character SHA-256 hexadecimal string.
        """
        hasher = hashlib.sha256()

        # Normalize state
        if isinstance(state, AgentState):
            # Canonical representation of state fields
            state_data = {
                "user_goal": state.user_goal.strip(),
                "environment": sorted(state.environment.items()),
                "recent_tools": [
                    {
                        "tool": t.tool,
                        "output_hash": t.output_hash or (
                            hashlib.sha256(t.output.encode("utf-8")).hexdigest()
                            if t.output else None
                        ),
                        "success": t.success,
                    }
                    for t in state.recent_tools
                ],
                "extra": (state.extra_context or "").strip(),
            }
            canonical_state = json.dumps(state_data, sort_keys=True)
        else:
            canonical_state = state.strip()

        hasher.update(canonical_state.encode("utf-8"))

        # Incorporate tool criteria schema so registry changes invalidate cache
        if criteria_schema:
            criteria_canonical = json.dumps(sorted(criteria_schema.items()), sort_keys=True)
            hasher.update(criteria_canonical.encode("utf-8"))

        return hasher.hexdigest()

    def get(self, state_hash: str) -> Optional[DecisionResult]:
        """Look up a cached decision by state hash.

        Returns None if cache is disabled, key not found, or entry expired.
        """
        if not self.enabled:
            return None

        entry = self._cache.get(state_hash)
        if entry is None:
            self.stats.misses += 1
            return None

        # Check TTL expiration
        now = time.time()
        if now - entry["timestamp"] > self.ttl_seconds:
            # Expired
            del self._cache[state_hash]
            self.stats.misses += 1
            return None

        # Cache hit: Move to end (most recently used)
        self._cache.move_to_end(state_hash)
        self.stats.hits += 1

        cached_decision: DecisionResult = entry["decision"]
        # Estimate latency saved from original execution time
        self.stats.total_latency_saved_ms += cached_decision.execution_time_ms or 150.0

        # Return clone with status CACHE_HIT and source 'cache'
        result_dict = cached_decision.model_dump()
        result_dict["status"] = DecisionStatus.CACHE_HIT
        result_dict["source"] = "cache"
        result_dict["execution_time_ms"] = 0.05  # sub-millisecond cache hit

        return DecisionResult(**result_dict)

    def set(self, state_hash: str, decision: DecisionResult) -> None:
        """Store a decision result in the cache."""
        if not self.enabled:
            return

        # Do not cache declined decisions or errors
        if decision.status == DecisionStatus.DECLINED:
            return

        # Evict oldest entry if capacity reached
        if len(self._cache) >= self.max_size and state_hash not in self._cache:
            self._cache.popitem(last=False)
            self.stats.evictions += 1

        self._cache[state_hash] = {
            "decision": decision,
            "timestamp": time.time(),
        }
        self._cache.move_to_end(state_hash)

    def invalidate(self, state_hash: str) -> bool:
        """Remove a specific entry from cache."""
        if state_hash in self._cache:
            del self._cache[state_hash]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()

    def size(self) -> int:
        """Current number of active entries."""
        return len(self._cache)
