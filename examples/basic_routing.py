"""Basic routing example demonstrating registry, fast-path cache, and confidence gating."""

from agent_decision_router import (
    DecisionEngine,
    ToolRegistry,
    RouteRequest,
    MockBackend,
)


def main():
    # 1. Initialize registry and custom tools
    registry = ToolRegistry()

    @registry.tool(
        criteria="Run unit tests, pytest, linter, or build pipelines",
        is_destructive=False,
    )
    def run_tests(suite: str) -> str:
        """Run project test suite."""
        return f"Ran tests for {suite}"

    @registry.tool(
        criteria="Inspect source code or configuration files",
        is_destructive=False,
    )
    def view_file(path: str) -> str:
        """View file contents."""
        return f"Contents of {path}"

    @registry.tool(
        criteria="Apply code edits, fix assertion errors, or write patches",
        is_destructive=True,
    )
    def edit_code(path: str, diff: str) -> str:
        """Apply patch to file."""
        return f"Patched {path}"

    # 2. Initialize Decision Engine with backend
    # In production, uses CloudflareBackend (@cf/cloudflare/clef-flash) or OllamaBackend
    engine = DecisionEngine(
        registry=registry,
        backend=MockBackend(canned_action="run_tests", canned_confidence=0.89),
    )

    state_prompt = "User requested: Run the unit tests and fix any assertion errors in auth_test.py. Current git status: 1 modified file."

    # 3. Route decision
    print("--- 1. First Routing Call (Cold / Cache Miss) ---")
    decision1 = engine.route_sync(RouteRequest(state=state_prompt))
    print(f"Action:       {decision1.action}")
    print(f"Status:       {decision1.status.value}")
    print(f"Confidence:   {decision1.confidence}")
    print(f"Source:       {decision1.source}")
    print(f"Latency:      {decision1.execution_time_ms} ms")

    # 4. Instant Fast-Path Cache hit on identical state
    print("\n--- 2. Second Routing Call (Stage 1 Fast-Path Cache Hit) ---")
    decision2 = engine.route_sync(RouteRequest(state=state_prompt))
    print(f"Action:       {decision2.action}")
    print(f"Status:       {decision2.status.value}")
    print(f"Source:       {decision2.source}")
    print(f"Latency:      {decision2.execution_time_ms} ms")

    # 5. Weak match refusal (confidence < 0.40)
    print("\n--- 3. Low Confidence Ambiguous Prompt (Declined) ---")
    weak_engine = DecisionEngine(
        registry=registry,
        backend=MockBackend(canned_action="run_tests", canned_confidence=0.28),
    )
    decision3 = weak_engine.route_sync(RouteRequest(state="Maybe do something somewhere in the code"))
    print(f"Action:       {decision3.action}")
    print(f"Status:       {decision3.status.value}")
    print(f"Confidence:   {decision3.confidence}")
    print(f"Reason:       {decision3.reason}")


if __name__ == "__main__":
    main()
