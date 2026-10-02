"""Command Line Interface for agent-decision-router."""

from __future__ import annotations

import json
import sys
import click
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agent_decision_router.config import get_settings
from agent_decision_router.models import RouteRequest, DecisionStatus
from agent_decision_router.registry import create_default_coding_registry
from agent_decision_router.engine import DecisionEngine
from agent_decision_router.backend import get_backend

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="agent-decision-router")
def main() -> None:
    """agent-decision-router: Confidence-aware tool & skill decision layer for AI coding agents."""
    pass


@main.command()
@click.option("--host", default=None, help="Host address to bind the API server")
@click.option("--port", default=None, type=int, help="Port to bind the API server")
@click.option("--reload", is_flag=True, help="Enable auto-reload on code change")
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the standalone FastAPI REST server."""
    settings = get_settings()
    server_host = host or settings.router_host
    server_port = port or settings.router_port

    console.print(
        Panel.fit(
            f"[bold green]agent-decision-router API Server[/bold green]\n"
            f"Host: [cyan]{server_host}:{server_port}[/cyan]\n"
            f"Backend: [yellow]{settings.router_backend}[/yellow]\n"
            f"Documentation: [link=http://{server_host}:{server_port}/docs]http://{server_host}:{server_port}/docs[/link]",
            title="Starting Server",
            border_style="green",
        )
    )
    uvicorn.run(
        "agent_decision_router.api:app",
        host=server_host,
        port=server_port,
        reload=reload,
    )


@main.command()
@click.option("--transport", default="stdio", type=click.Choice(["stdio", "sse"]), help="MCP transport protocol")
def mcp(transport: str) -> None:
    """Launch the FastMCP server for Claude Code, Cline, Cursor, etc."""
    from agent_decision_router.mcp_server import run_server
    console.print(f"[bold cyan]Launching FastMCP server over {transport}...[/bold cyan]", file=sys.stderr)
    run_server(transport=transport)


@main.command()
@click.argument("state", type=str)
@click.option("--backend", "-b", default=None, help="Backend provider override (cloudflare, ollama, mock)")
@click.option("--skip-cache", is_flag=True, help="Bypass Stage 1 Fast-Path Cache")
def route(state: str, backend: str | None, skip_cache: bool) -> None:
    """Route a single decision for the given state description."""
    engine = DecisionEngine(
        registry=create_default_coding_registry(),
        backend=get_backend(name=backend),
    )

    console.print(f"[bold]Evaluating State:[/bold] {state}")
    result = engine.route_sync(RouteRequest(state=state, skip_cache=skip_cache))

    # Format result status color
    status_colors = {
        DecisionStatus.EXECUTABLE: "bold green",
        DecisionStatus.CACHE_HIT: "bold cyan",
        DecisionStatus.LOOKAHEAD_RESOLVED: "bold yellow",
        DecisionStatus.DECLINED: "bold red",
        DecisionStatus.TASK_COMPLETED: "bold magenta",
    }
    color = status_colors.get(result.status, "white")

    table = Table(title="Routing Decision Result", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Action", f"[{color}]{result.action}[/{color}]")
    table.add_row("Status", f"[{color}]{result.status.value}[/{color}]")
    table.add_row("Confidence", f"{result.confidence:.3f}")
    table.add_row("Source", result.source)
    table.add_row("Destructive?", str(result.is_destructive))
    table.add_row("Requires Confirmation?", str(result.requires_confirmation))
    table.add_row("Task Completion?", str(result.task_completion))
    table.add_row("Latency", f"{result.execution_time_ms:.2f} ms")
    if result.reason:
        table.add_row("Reason", result.reason)

    console.print(table)

    if result.probabilities:
        prob_table = Table(title="Candidate Probabilities", show_header=True)
        prob_table.add_column("Candidate Action", style="cyan")
        prob_table.add_column("Probability", style="yellow")
        for act, prob in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
            prob_table.add_row(act, f"{prob:.4f}")
        console.print(prob_table)


@main.command()
def tools() -> None:
    """List standard registered tools and criteria."""
    registry = create_default_coding_registry()
    table = Table(title="Registered Tools & Routing Criteria", show_header=True)
    table.add_column("Tool Name", style="bold cyan")
    table.add_column("Criteria (Decision Key)", style="green")
    table.add_column("Destructive", style="red")

    for tool in registry.list_tools():
        table.add_row(
            tool.name,
            tool.criteria,
            "[red]Yes[/red]" if tool.is_destructive else "[dim]No[/dim]",
        )
    console.print(table)


if __name__ == "__main__":
    main()
