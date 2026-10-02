"""Data models and schemas for agent-decision-router."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Supported question types for System 1 decision backends."""
    CHOICE = "choice"
    NOUL = "noul"
    TEXT = "text"


class DecisionStatus(str, Enum):
    """Routing outcome status based on confidence gating."""
    EXECUTABLE = "EXECUTABLE"
    LOOKAHEAD_RESOLVED = "LOOKAHEAD_RESOLVED"
    DECLINED = "DECLINED"
    TASK_COMPLETED = "TASK_COMPLETED"
    CACHE_HIT = "CACHE_HIT"


class ToolDefinition(BaseModel):
    """Definition of an agent tool with routing criteria."""
    name: str = Field(..., description="Unique tool name (e.g., 'run_terminal_command')")
    description: str = Field(..., description="Human-readable description of tool capabilities")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema or parameter specification for the tool"
    )
    criteria: str = Field(
        ...,
        description="Routing criteria explaining when this tool should be selected"
    )
    is_destructive: bool = Field(
        default=False,
        description="Whether this tool potentially mutates files, terminates processes, or drops data"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional metadata"
    )


class SkillDefinition(BaseModel):
    """High-level skill or workflow composed of one or more tools."""
    name: str = Field(..., description="Unique skill name")
    description: str = Field(..., description="Skill description")
    criteria: str = Field(..., description="Routing criteria for triggering this skill")
    tools: List[str] = Field(
        default_factory=list,
        description="Tools participating in this skill"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary skill metadata"
    )


class QuestionDefinition(BaseModel):
    """Typed question definition sent to the decision model."""
    type: str = Field(..., description="Question type ('choice', 'noul', etc.)")
    instructions: str = Field(..., description="Instructions guiding the decision")
    criteria: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping of choice candidate keys to criteria descriptions (for choice type)"
    )


class DecisionPayload(BaseModel):
    """REST request payload format for decision backends."""
    state: str = Field(..., description="Stringified context and environmental state")
    questions: Dict[str, QuestionDefinition] = Field(
        ...,
        description="Dictionary of typed questions (next_tool, is_destructive, task_completion)"
    )


class DecisionResult(BaseModel):
    """Result returned by the decision engine pipeline."""
    action: str = Field(..., description="Selected tool/skill name, 'DECLINE', or 'COMPLETE_TASK'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the decision")
    status: DecisionStatus = Field(..., description="Routing pipeline status")
    is_destructive: bool = Field(
        default=False,
        description="True if the chosen action is predicted to be destructive"
    )
    destructive_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that the action is destructive"
    )
    task_completion: bool = Field(
        default=False,
        description="True if user task is judged complete and agent should stop"
    )
    task_completion_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that task is complete"
    )
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across candidate actions"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanation or rationale (especially when declined or guarded)"
    )
    source: str = Field(
        default="system1",
        description="Source of the decision ('cache', 'system1', 'lookahead', 'fallback')"
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Execution latency in milliseconds"
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Whether agent runtime should pause for human approval before executing"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional debugging or routing telemetry"
    )


class ToolCallRecord(BaseModel):
    """Record of a recently executed tool call in agent trajectory."""
    tool: str = Field(..., description="Name of the tool executed")
    arguments: Optional[Dict[str, Any]] = Field(default=None, description="Arguments passed to the tool")
    output: Optional[str] = Field(default=None, description="Tool output summary or content")
    output_hash: Optional[str] = Field(default=None, description="SHA-256 hash of tool output")
    success: bool = Field(default=True, description="Whether tool execution succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class AgentState(BaseModel):
    """Structured representation of agent environment and task context."""
    user_goal: str = Field(..., description="Original user prompt or task goal")
    recent_tools: List[ToolCallRecord] = Field(
        default_factory=list,
        description="List of recently executed tool calls and outputs"
    )
    environment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment context: git status, open files, error trace, etc."
    )
    extra_context: Optional[str] = Field(
        default=None,
        description="Any supplementary context or user instructions"
    )

    def to_state_string(self) -> str:
        """Convert structured agent state to concise string format for System 1 prompt."""
        parts = [f"User Goal: {self.user_goal.strip()}"]

        if self.environment:
            env_items = [f"{k}: {v}" for k, v in self.environment.items()]
            parts.append(f"Environment: {', '.join(env_items)}")

        if self.recent_tools:
            tool_history = []
            for t in self.recent_tools[-5:]:  # Keep recent 5 tool calls
                status = "success" if t.success else f"failed: {t.error}"
                out_summary = f" (output: {t.output[:100]}...)" if t.output else ""
                tool_history.append(f"- {t.tool} [{status}]{out_summary}")
            parts.append("Recent Tool Calls:\n" + "\n".join(tool_history))

        if self.extra_context:
            parts.append(f"Context: {self.extra_context.strip()}")

        return "\n\n".join(parts)


class RouteRequest(BaseModel):
    """API and MCP request payload for routing the next action."""
    state: Union[str, AgentState] = Field(
        ...,
        description="State string or structured AgentState representation"
    )
    candidate_tools: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of tool names to consider"
    )
    skip_cache: bool = Field(
        default=False,
        description="Bypass fast-path cache if true"
    )
    confidence_threshold_high: Optional[float] = Field(
        default=None,
        description="Override threshold for immediate execution (default 0.70)"
    )
    confidence_threshold_low: Optional[float] = Field(
        default=None,
        description="Override threshold for declining weak matches (default 0.40)"
    )


class RouteResponse(BaseModel):
    """API response model."""
    success: bool = True
    decision: DecisionResult
    meta: Dict[str, Any] = Field(default_factory=dict)
