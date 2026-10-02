"""agent-decision-router: A framework-agnostic tool & skill decision layer for AI coding agents."""

from agent_decision_router.models import (
    ToolDefinition,
    SkillDefinition,
    QuestionDefinition,
    QuestionType,
    DecisionResult,
    DecisionStatus,
    ToolCallRecord,
    AgentState,
    RouteRequest,
    RouteResponse,
)
from agent_decision_router.registry import (
    ToolRegistry,
    create_default_coding_registry,
)
from agent_decision_router.cache import FastPathCache, CacheStats
from agent_decision_router.lookahead import LookaheadEvaluator, MCTSNode
from agent_decision_router.engine import DecisionEngine
from agent_decision_router.backend import (
    BaseDecisionBackend,
    ParsedDecision,
    CloudflareBackend,
    OllamaBackend,
    MockBackend,
    get_backend,
)
from agent_decision_router.config import get_settings, Settings

__version__ = "0.1.0"

__all__ = [
    "ToolDefinition",
    "SkillDefinition",
    "QuestionDefinition",
    "QuestionType",
    "DecisionResult",
    "DecisionStatus",
    "ToolCallRecord",
    "AgentState",
    "RouteRequest",
    "RouteResponse",
    "ToolRegistry",
    "create_default_coding_registry",
    "FastPathCache",
    "CacheStats",
    "LookaheadEvaluator",
    "MCTSNode",
    "DecisionEngine",
    "BaseDecisionBackend",
    "ParsedDecision",
    "CloudflareBackend",
    "OllamaBackend",
    "MockBackend",
    "get_backend",
    "get_settings",
    "Settings",
]
