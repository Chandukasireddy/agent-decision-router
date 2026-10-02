"""Full coding agent loop demonstration with PreDecision/PostTool lifecycle."""

from agent_decision_router import (
    DecisionEngine,
    AgentState,
    DecisionStatus,
    MockBackend,
    create_default_coding_registry,
)


def main():
    registry = create_default_coding_registry()
    engine = DecisionEngine(
        registry=registry,
        backend=MockBackend(auto_heuristic=True),
    )

    # Initialize agent state
    state = AgentState(
        user_goal="Run unit tests and fix any assertion errors in auth_test.py",
        environment={"git_status": "clean", "repo": "agent-decision-router"},
    )

    # Setup PostTool hook to track trajectory telemetry
    def on_tool_executed(record, agent_state):
        print(f"  [PostTool Hook] Recorded '{record.tool}' -> Success: {record.success}")

    engine.add_post_tool_hook(on_tool_executed)

    print("=== Step 1: Initial Action Selection ===")
    decision = engine.route_sync(state)
    print(f"Decision: {decision.action} (Status: {decision.status.value}, Confidence: {decision.confidence:.2f})")

    # Simulate tool execution (pytest fails)
    engine.record_tool_result(
        tool="run_terminal_command",
        arguments={"command": "pytest tests/auth_test.py"},
        output="FAILED tests/auth_test.py::test_login - AssertionError: 401 != 200",
        success=False,
        error="AssertionError in auth_test.py",
        state=state,
    )

    print("\n=== Step 2: Next Action After Test Failure ===")
    decision = engine.route_sync(state)
    print(f"Decision: {decision.action} (Status: {decision.status.value}, Confidence: {decision.confidence:.2f})")

    # Simulate reading file
    engine.record_tool_result(
        tool="read_code_file",
        arguments={"path": "auth_test.py"},
        output="def test_login(): assert response.status_code == 200",
        success=True,
        state=state,
    )

    print("\n=== Step 3: Next Action After Inspecting Code ===")
    decision = engine.route_sync(state)
    print(f"Decision: {decision.action} (Status: {decision.status.value}, Requires Confirmation: {decision.requires_confirmation})")

    # Simulate patch and passing tests
    engine.record_tool_result(
        tool="apply_code_patch",
        arguments={"path": "auth.py", "content": "return 200"},
        output="Patch applied successfully",
        success=True,
        state=state,
    )
    engine.record_tool_result(
        tool="run_terminal_command",
        arguments={"command": "pytest"},
        output="32 passed in 1.4s. All tests passed successfully.",
        success=True,
        state=state,
    )

    print("\n=== Step 4: Verification & Task Completion ===")
    state.extra_context = "All tests passed and fix verified. Objective satisfied."
    decision = engine.route_sync(state)
    print(f"Decision: {decision.action} (Status: {decision.status.value}, Task Complete: {decision.task_completion})")
    print(f"Reason:   {decision.reason}")


if __name__ == "__main__":
    main()
