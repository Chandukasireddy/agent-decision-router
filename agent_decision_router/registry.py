"""Dynamic Tool, Skill, and AI Model Registry with criteria schema generation."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from agent_decision_router.models import (
    ToolDefinition,
    SkillDefinition,
    ModelTierDefinition,
    QuestionDefinition,
    QuestionType,
)


class ToolRegistry:
    """Dynamic registry for tools, skills, and model tiers with criteria generation for System 1 routing."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._skills: Dict[str, SkillDefinition] = {}
        self._model_tiers: Dict[str, ModelTierDefinition] = {}
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
        """Register a tool with its routing criteria."""
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
        """Register a high-level skill with its routing criteria."""
        skill = SkillDefinition(
            name=name,
            description=description,
            criteria=criteria,
            tools=tools or [],
            metadata=metadata or {},
        )
        self._skills[name] = skill
        return skill

    def register_model_tier(
        self,
        name: str,
        provider: str,
        description: str,
        criteria: str,
        context_window: str = "128k",
        cost_tier: str = "medium",
        specialty: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelTierDefinition:
        """Register a publicly available AI model with routing criteria."""
        tier = ModelTierDefinition(
            name=name,
            provider=provider,
            description=description,
            criteria=criteria,
            context_window=context_window,
            cost_tier=cost_tier,
            specialty=specialty,
            metadata=metadata or {},
        )
        self._model_tiers[name] = tier
        return tier

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

    def get_model_tier(self, name: str) -> Optional[ModelTierDefinition]:
        """Get model definition by name."""
        return self._model_tiers.get(name)

    def get_callable(self, name: str) -> Optional[Callable[..., Any]]:
        """Get underlying executable function if registered."""
        return self._callables.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Return list of all registered tools."""
        return list(self._tools.values())

    def list_skills(self) -> List[SkillDefinition]:
        """Return list of all registered skills."""
        return list(self._skills.values())

    def list_model_tiers(self) -> List[ModelTierDefinition]:
        """Return list of all registered models."""
        return list(self._model_tiers.values())

    def unregister(self, name: str) -> bool:
        """Remove a tool, skill, or model from the registry."""
        removed = False
        if name in self._tools:
            del self._tools[name]
            removed = True
        if name in self._skills:
            del self._skills[name]
            removed = True
        if name in self._model_tiers:
            del self._model_tiers[name]
            removed = True
        if name in self._callables:
            del self._callables[name]
        return removed

    def clear(self) -> None:
        """Clear all registered tools, skills, and models."""
        self._tools.clear()
        self._skills.clear()
        self._model_tiers.clear()
        self._callables.clear()

    def generate_model_criteria_schema(self) -> Dict[str, str]:
        """Generate criteria schema for model choice question."""
        return {name: tier.criteria for name, tier in self._model_tiers.items()}

    def generate_skill_criteria_schema(self) -> Dict[str, str]:
        """Generate criteria schema for skill choice question."""
        return {name: skill.criteria for name, skill in self._skills.items()}

    def generate_criteria_schema(
        self,
        candidate_names: Optional[List[str]] = None,
        include_skills: bool = True,
    ) -> Dict[str, str]:
        """Generate the JSON criteria dictionary for the decision model."""
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
        include_models: bool = True,
        include_skills_question: bool = True,
        next_action_instructions: str = "Which tool or action should the agent execute next?",
        is_destructive_instructions: str = "Will this planned action delete data, overwrite uncommitted changes, or terminate processes?",
        task_completion_instructions: str = "Is the requested task already completed and fulfilled? (Answer false if tools or actions still need to be performed)",
    ) -> Dict[str, QuestionDefinition]:
        """Generate the standard typed questions dictionary for System 1 routing."""
        criteria = self.generate_criteria_schema(
            candidate_names=candidate_names,
            include_skills=include_skills,
        )

        questions: Dict[str, QuestionDefinition] = {}

        # 1. Model Selection (Auto Model Selector)
        if include_models and self._model_tiers:
            questions["model_tier"] = QuestionDefinition(
                type=QuestionType.CHOICE.value,
                instructions="Which AI model is best suited and most cost-effective for this task?",
                criteria=self.generate_model_criteria_schema(),
            )

        # 2. High-level Domain Skill Selection
        if include_skills_question and self._skills:
            questions["selected_skill"] = QuestionDefinition(
                type=QuestionType.CHOICE.value,
                instructions="Which specialized skill domain is most appropriate for this request?",
                criteria=self.generate_skill_criteria_schema(),
            )

        # 3. Next Tool Action Selection
        questions["next_tool"] = QuestionDefinition(
            type=QuestionType.CHOICE.value,
            instructions=next_action_instructions,
            criteria=criteria,
        )

        # 4. Safety Guard: Is Destructive?
        questions["is_destructive"] = QuestionDefinition(
            type=QuestionType.NOUL.value,
            instructions=is_destructive_instructions,
        )

        # 5. Task Completion Guard: Is Task Finished?
        questions["task_completion"] = QuestionDefinition(
            type=QuestionType.NOUL.value,
            instructions=task_completion_instructions,
        )

        return questions


def create_default_coding_registry() -> ToolRegistry:
    """Create a ToolRegistry populated with real publicly available models, skills, and tools."""
    registry = ToolRegistry()

    # --------------------------------------------------------------------------
    # 1. Publicly Available AI Models (Auto Model Selection)
    # --------------------------------------------------------------------------
    registry.register_model_tier(
        name="claude_3_5_sonnet",
        provider="Anthropic",
        description="Industry benchmark leader for coding, agentic reasoning, and complex tool-use.",
        criteria="Complex multi-file refactoring, architectural overhaul, nuanced bug fixing, and subtle algorithmic reasoning",
        context_window="200k",
        cost_tier="premium",
        specialty="SWE-bench SOTA & Complex Architecture",
        metadata={"input_per_m": "$3.00", "output_per_m": "$15.00", "latency": "medium"},
    )
    registry.register_model_tier(
        name="deepseek_r1",
        provider="DeepSeek",
        description="Frontier open reasoning model utilizing deep chain-of-thought verification.",
        criteria="Complex mathematical verification, race condition diagnosis, deep algorithmic logic, and hard reasoning",
        context_window="64k",
        cost_tier="low",
        specialty="Deep Reasoning & Algorithmic Logic",
        metadata={"input_per_m": "$0.55", "output_per_m": "$2.19", "latency": "deliberate"},
    )
    registry.register_model_tier(
        name="gpt_4o",
        provider="OpenAI",
        description="High-speed flagship multimodal model with strong function calling and visual reasoning.",
        criteria="Multimodal UI screenshot inspection, visual layout debugging, and general multifaceted tasks",
        context_window="128k",
        cost_tier="medium",
        specialty="Multimodal Vision & Function Calling",
        metadata={"input_per_m": "$2.50", "output_per_m": "$10.00", "latency": "fast"},
    )
    registry.register_model_tier(
        name="gemini_2_0_flash",
        provider="Google",
        description="Sub-second ultra-fast model with massive context window for fast iterations.",
        criteria="Running test suites, quick edits, high-speed iterations, git workflows, and large context search",
        context_window="1000k (1M)",
        cost_tier="low",
        specialty="Sub-second Latency & 1M Context",
        metadata={"input_per_m": "$0.10", "output_per_m": "$0.40", "latency": "ultra_fast"},
    )
    registry.register_model_tier(
        name="claude_3_5_haiku",
        provider="Anthropic",
        description="Fast, low-cost model optimized for single-line changes, typo fixes, and formatting.",
        criteria="Simple lookups, minor typo fixes, docstring formatting, greetings, or short single-step answers",
        context_window="200k",
        cost_tier="ultra_low",
        specialty="High-speed Edits & Low Cost",
        metadata={"input_per_m": "$0.80", "output_per_m": "$4.00", "latency": "ultra_fast"},
    )
    registry.register_model_tier(
        name="qwen_2_5_coder_32b",
        provider="Alibaba / Open",
        description="Specialized open code generation model rivaling proprietary models on coding benchmarks.",
        criteria="Targeted syntax transformations, boilerplate generation, and local offline coding",
        context_window="128k",
        cost_tier="ultra_low",
        specialty="Code Completion & Offline Autonomy",
        metadata={"input_per_m": "$0.20", "output_per_m": "$0.60", "latency": "fast"},
    )

    # --------------------------------------------------------------------------
    # 2. Publicly Recognized Agent Skills
    # --------------------------------------------------------------------------
    registry.register_skill(
        name="test_and_verification",
        description="Execute test suites (pytest/npm test), analyze stack traces, and verify error fixes",
        criteria="Run test suites, inspect test runner output, analyze failures, or verify bug fixes",
        tools=["run_terminal_command", "read_code_file"],
        metadata={"category": "Quality Assurance", "risk_level": "medium"},
    )
    registry.register_skill(
        name="architectural_refactoring",
        description="Overhaul multi-file structure, apply code patches, and implement design patterns",
        criteria="Modify existing codebase files, write code, apply patches, or refactor components",
        tools=["apply_code_patch", "read_code_file"],
        metadata={"category": "Core Engineering", "risk_level": "high"},
    )
    registry.register_skill(
        name="security_and_auth_audit",
        description="Inspect authentication tokens, audit timing attack risks, and review security policies",
        criteria="Security audit, authentication analysis, token verification, and vulnerability scanning",
        tools=["search_codebase", "read_code_file"],
        metadata={"category": "Security", "risk_level": "low"},
    )
    registry.register_skill(
        name="codebase_navigation",
        description="Search codebase for symbols, regex patterns, file locations, and inspect existing code",
        criteria="Search codebase for symbols, find file patterns, inspect file contents, or discover code structure",
        tools=["search_codebase", "read_code_file"],
        metadata={"category": "Exploration", "risk_level": "low"},
    )
    registry.register_skill(
        name="system_and_devops",
        description="Execute terminal commands, manage processes, install dependencies, or inspect OS environment",
        criteria="Execute terminal commands, manage processes, install dependencies, or inspect OS environment",
        tools=["run_terminal_command"],
        metadata={"category": "DevOps", "risk_level": "high"},
    )
    registry.register_skill(
        name="user_consultation",
        description="Ask clarifying questions, request user confirmation, or clarify ambiguous requirements",
        criteria="Clarify ambiguous user requests, ask follow-up questions, or request human confirmation",
        tools=["ask_user_clarification"],
        metadata={"category": "Interaction", "risk_level": "none"},
    )

    # --------------------------------------------------------------------------
    # 3. Granular Tool Actions
    # --------------------------------------------------------------------------
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
        is_destructive=True,
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
