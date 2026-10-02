"""MCP Client integration example.

Demonstrates how external coding agents (Claude Code, Cursor, Cline, Codex)
call the agent-decision-router MCP server tools.
"""

from agent_decision_router.mcp_server import (
    route_next_action,
    register_agent_tool,
    list_agent_tools,
    get_router_cache_stats,
)


def main():
    print("--- 1. List Available Agent Tools via MCP ---")
    tools = list_agent_tools()
    for t in tools:
        print(f"  - {t['name']}: {t['criteria']} (Destructive: {t['is_destructive']})")

    print("\n--- 2. Register Custom Tool via MCP ---")
    reg = register_agent_tool(
        name="run_database_migration",
        description="Apply Alembic or Prisma migrations to local database",
        criteria="Run database schema migration, alembic upgrade, or prisma db push",
        is_destructive=True,
    )
    print(f"Registered tool: {reg['tool']['name']}")

    print("\n--- 3. Call route_next_action ---")
    state_description = (
        "User says: Run alembic upgrade head to apply pending database migrations. "
        "Current branch: feature/user-profiles."
    )
    result = route_next_action(
        state=state_description,
        skip_cache=False,
    )

    print(f"Routed Action:          {result['action']}")
    print(f"Status:                 {result['status']}")
    print(f"Confidence:             {result['confidence']}")
    print(f"Is Destructive:         {result['is_destructive']}")
    print(f"Requires Confirmation:  {result['requires_confirmation']}")

    print("\n--- 4. Call route_next_action again (Fast-Path Cache Hit) ---")
    result2 = route_next_action(
        state=state_description,
        skip_cache=False,
    )
    print(f"Routed Action:          {result2['action']}")
    print(f"Status:                 {result2['status']}")
    print(f"Source:                 {result2['source']}")

    print("\n--- 5. Cache Statistics ---")
    stats = get_router_cache_stats()
    print(f"Cache Hits:             {stats['hits']}")
    print(f"Cache Misses:           {stats['misses']}")
    print(f"Cache Hit Rate:         {stats['hit_rate']}")
    print(f"Cache Size:             {stats['current_cache_size']}")


if __name__ == "__main__":
    main()
