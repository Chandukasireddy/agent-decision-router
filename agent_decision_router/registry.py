"""Dynamic Tool and Skill Registry with criteria schema generation."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from agent_decision_router.models import (
    ToolDefinition,
    SkillDefinition,
    QuestionDefinition,
    QuestionType,
)


class ToolRegistry:
    """Dynamic registry for tools and skills with criteria generation for System 1 routing."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._skills: Dict[str, SkillDefinition] = {}
        self._callables: Dict[str, Callable[..., Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        criteria: str,
        parameters: Optional[Dict[str, Any]] = None,
        is_destructive: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        func: Optional[Callable[..., Any]] = None,
    ) -> ToolDefinition:
        """Register a tool with its routing criteria.

        Args:
            name: Unique name identifier for the tool.
            description: Description of what the tool does.
            criteria: Routing criteria explaining when the model should select this tool.
            parameters: Optional parameter schema.
            is_destructive: Whether this tool performs destructive actions (file deletion, git drop, etc.).
            metadata: Optional additional metadata.
            func: Optional underlying callable implementation.

        Returns:
            The created ToolDefinition.
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
            criteria=criteria,
            is_destructive=is_destructive,
            metadata=metadata or {},
        )
        self._tools[name] = tool
        if func is not None:
            self._callables[name] = func
        return tool

    def register_skill(
        self,
        name: str,
        description: str,
        criteria: str,
        tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillDefinition:
        """Register a high-level skill with its routing criteria.

        Args:
            name: Unique skill name identifier.
            description: Description of what the skill encompasses.
            criteria: Routing criteria explaining when this skill should be activated.
            tools: Names of tools belonging to this skill.
            metadata: Optional additional metadata.

        Returns:
            The created SkillDefinition.
        """
        skill = SkillDefinition(
            name=name,
            description=description,
            criteria=criteria,
            tools=tools or [],
            metadata=metadata or {},
        )
        self._skills[name] = skill
        return skill

    def tool(
        self,
        name: Optional[str] = None,
        criteria: Optional[str] = None,
        description: Optional[str] = None,
        is_destructive: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a tool in the registry."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip() or f"Execute {tool_name}"
            tool_criteria = criteria or tool_desc

            # Inspect signature to generate basic parameter schema if not provided
            sig = inspect.signature(func)
            params: Dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            for p_name, p in sig.parameters.items():
                if p_name in ("self", "cls"):
                    continue
                param_type = "string"
                if p.annotation == int:
                    param_type = "integer"
                elif p.annotation == float:
                    param_type = "number"
                elif p.annotation == bool:
                    param_type = "boolean"
                elif p.annotation in (dict, Dict):
                    param_type = "object"
                elif p.annotation in (list, List):
                    param_type = "array"

                params["properties"][p_name] = {"type": param_type}
                if p.default == inspect.Parameter.empty:
                    params["required"].append(p_name)

            self.register_tool(
                name=tool_name,
                description=tool_desc,
                criteria=tool_criteria,
                parameters=params,
                is_destructive=is_destructive,
                metadata=metadata,
                func=func,
            )
            return func
        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return self._tools.get(name)

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Get skill definition by name."""
        return self._skills.get(name)

    def get_callable(self, name: str) -> Optional[Callable[..., Any]]:
        """Get underlying executable function if registered."""
        return self._callables.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Return list of all registered tools."""
        return list(self._tools.values())

    def list_skills(self) -> List[SkillDefinition]:
        """Return list of all registered skills."""
        return list(self._skills.values())

    def unregister(self, name: str) -> bool:
        """Remove a tool or skill from the registry."""
        removed = False
        if name in self._tools:
            del self._tools[name]
            removed = True
        if name in self._skills:
            del self._skills[name]
            removed = True
        if name in self._callables:
            del self._callables[name]
        return removed

    def clear(self) -> None:
        """Clear all registered tools and skills."""
        self._tools.clear__(None) if hasattr(self._tools, "clear__") else self._tools.clear()
        self._skills.clear()
        self._callables.clear()

    def generate_criteria_schema(
        self,
        candidate_names: Optional[List[str]] = None,
        include_skills: bool = True,
    ) -> Dict[str, str]:
        """Generate the JSON criteria dictionary for the decision model.

        Maps candidate keys to their descriptive criteria strings.

        Args:
            candidate_names: Optional whitelist of tool/skill names to include.
            include_skills: Whether to include registered skills alongside tools.

        Returns:
            Dict[str, str]: Mapping of tool/skill name -> criteria string.
        """
        criteria_map: Dict[str, str] = {}

        # Add tools
        for tool_name, tool in self._tools.items():
            if candidate_names is None or tool_name in candidate_names:
                criteria_map[tool_name] = tool.criteria

        # Add skills if requested
        if include_skills:
            for skill_name, skill in self._skills.items():
                if candidate_names is None or skill_name in candidate_names:
                    criteria_map[skill_name] = skill.criteria

        return criteria_map

    def generate_questions(
        self,
        candidate_names: Optional[List[str]] = None,
        include_skills: bool = True,
        next_action_instructions: str = "Which tool or skill should the agent run next?",
        is_destructive_instructions: str = "Will this planned action delete data, overwrite uncommitted changes, or terminate processes?",
        task_completion_instructions: str = "Is the requested task already completed and fulfilled? (Answer false if tools or actions still need to be performed)",
    ) -> Dict[str, QuestionDefinition]:
        """Generate the standard typed questions dictionary for System 1 routing."""
        criteria = self.generate_criteria_schema(
            candidate_names=candidate_names,
            include_skills=include_skills,
        )

        return {
            "next_tool": QuestionDefinition(
                type=QuestionType.CHOICE.value,
                instructions=next_action_instructions,
                criteria=criteria,
            ),
            "is_destructive": QuestionDefinition(
                type=QuestionType.NOUL.value,
                instructions=is_destructive_instructions,
            ),
            "task_completion": QuestionDefinition(
                type=QuestionType.NOUL.value,
                instructions=task_completion_instructions,
            ),
        }


def create_default_coding_registry() -> ToolRegistry:
    """Create a ToolRegistry populated with standard coding agent tools."""
    registry = ToolRegistry()

    registry.register_tool(
        name="run_terminal_command",
        description="Execute shell commands, run test runners (pytest/npm test), linters, or build scripts.",
        criteria="Execute shell commands, pytest, test runners, or build tools",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line string to execute"}
            },
            "required": ["command"]
        },
        is_destructive=True,  # Running terminal commands may be destructive
    )

    registry.register_tool(
        name="read_code_file",
        description="Inspect contents of a file on disk or view lines within a specific range.",
        criteria="Inspect contents of a file on disk",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to inspect"}
            },
            "required": ["path"]
        },
        is_destructive=False,
    )

    registry.register_tool(
        name="apply_code_patch",
        description="Write new files or apply edits and patches to existing source code.",
        criteria="Write or modify existing source files",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the target file"},
                "content": {"type": "string", "description": "New content or patch"}
            },
            "required": ["path"]
        },
        is_destructive=True,
    )

    registry.register_tool(
        name="search_codebase",
        description="Search for regex patterns, symbol definitions, or text occurrences across workspace files.",
        criteria="Search for file patterns, symbols, or text across the project directory",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query or regex to search for"}
            },
            "required": ["query"]
        },
        is_destructive=False,
    )

    registry.register_tool(
        name="ask_user_clarification",
        description="Prompt the user for additional details, confirmations, or clarify ambiguous requirements.",
        criteria="Task is ambiguous, missing required configuration, or requires human input",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Clarification question for the user"}
            },
            "required": ["question"]
        },
        is_destructive=False,
    )

    return registry
