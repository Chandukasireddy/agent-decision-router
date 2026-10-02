"""Tests for ToolRegistry and criteria schema generation."""

import pytest
from agent_decision_router.registry import ToolRegistry, create_default_coding_registry
from agent_decision_router.models import QuestionType


def test_registry_registration_and_lookup():
    registry = ToolRegistry()

    tool = registry.register_tool(
        name="test_tool",
        description="A test tool",
        criteria="Execute test action when asked",
        parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        is_destructive=True,
    )

    assert tool.name == "test_tool"
    assert tool.is_destructive is True
    assert registry.get_tool("test_tool") is not None
    assert registry.get_tool("nonexistent") is None
    assert len(registry.list_tools()) == 1


def test_registry_skill_registration():
    registry = ToolRegistry()

    skill = registry.register_skill(
        name="refactor_module",
        description="Refactor entire code module",
        criteria="User asks to refactor, restructure, or clean up an entire module",
        tools=["read_code_file", "apply_code_patch"],
    )

    assert skill.name == "refactor_module"
    assert skill.tools == ["read_code_file", "apply_code_patch"]
    assert registry.get_skill("refactor_module") is not None


def test_registry_decorator():
    registry = ToolRegistry()

    @registry.tool(
        criteria="Format Python code with black or ruff",
        is_destructive=True,
    )
    def format_code(filepath: str, line_length: int = 88) -> str:
        """Run code formatter on target file."""
        return f"formatted {filepath}"

    tool = registry.get_tool("format_code")
    assert tool is not None
    assert tool.criteria == "Format Python code with black or ruff"
    assert tool.is_destructive is True
    assert "filepath" in tool.parameters["properties"]
    assert "line_length" in tool.parameters["properties"]
    assert tool.parameters["required"] == ["filepath"]


def test_criteria_schema_generation():
    registry = create_default_coding_registry()
    criteria_map = registry.generate_criteria_schema()

    assert "run_terminal_command" in criteria_map
    assert "read_code_file" in criteria_map
    assert "apply_code_patch" in criteria_map
    assert "search_codebase" in criteria_map
    assert "ask_user_clarification" in criteria_map

    # Test candidate filtering
    filtered = registry.generate_criteria_schema(candidate_names=["read_code_file", "apply_code_patch"])
    assert len(filtered) == 2
    assert "read_code_file" in filtered
    assert "apply_code_patch" in filtered
    assert "run_terminal_command" not in filtered


def test_questions_generation_payload():
    registry = create_default_coding_registry()
    questions = registry.generate_questions()

    assert "next_tool" in questions
    assert "is_destructive" in questions
    assert "task_completion" in questions

    assert questions["next_tool"].type == QuestionType.CHOICE.value
    assert questions["is_destructive"].type == QuestionType.NOUL.value
    assert questions["task_completion"].type == QuestionType.NOUL.value
    assert isinstance(questions["next_tool"].criteria, dict)
    assert len(questions["next_tool"].criteria) >= 5
