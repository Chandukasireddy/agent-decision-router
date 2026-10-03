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
            f"Interactive Dashboard: [link=http://{server_host}:{server_port}]http://{server_host}:{server_port}[/link]\n"
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


def _render_prob_bar(prob: float, width: int = 18) -> str:
    """Render a colored ASCII probability bar safe for all Windows consoles."""
    filled = max(0, min(width, int(round(prob * width))))
    empty = width - filled
    if prob >= 0.70:
        bar_color = "bold green"
    elif prob >= 0.40:
        bar_color = "bold yellow"
    else:
        bar_color = "dim cyan"
    bar_chars = "=" * filled + "-" * empty
    return f"[{bar_color}][{bar_chars}][/{bar_color}] [bold]{prob * 100:5.1f}%[/bold]"


@main.command()
@click.argument("state", type=str)
@click.option("--backend", "-b", default=None, help="Backend provider override (cloudflare, ollama, mock)")
@click.option("--skip-cache", is_flag=True, help="Bypass Stage 1 Fast-Path Cache")
@click.option("--json", "-j", "output_json", is_flag=True, help="Output decision as raw JSON")
def route(state: str, backend: str | None, skip_cache: bool, output_json: bool) -> None:
    """Route a decision for the given user prompt or agent state."""
    registry = create_default_coding_registry()
    engine = DecisionEngine(
        registry=registry,
        backend=get_backend(name=backend),
    )

    result = engine.route_sync(RouteRequest(state=state, skip_cache=skip_cache))

    if output_json:
        console.print(result.model_dump_json(indent=2))
        return

    # --------------------------------------------------------------------------
    # Output Stage: Visual 4-Layer Terminal Display
    # --------------------------------------------------------------------------
    status_styles = {
        DecisionStatus.EXECUTABLE: ("EXECUTABLE (Auto-Approved)", "bold green"),
        DecisionStatus.CACHE_HIT: ("CACHE HIT (Zero-Latency)", "bold cyan"),
        DecisionStatus.LOOKAHEAD_RESOLVED: ("LOOKAHEAD RESOLVED (MCTS Rollout)", "bold yellow"),
        DecisionStatus.DECLINED: ("DECLINED (Refused Weak Match)", "bold red"),
        DecisionStatus.TASK_COMPLETED: ("TASK COMPLETED (Agent Stop)", "bold magenta"),
    }
    status_label, status_color = status_styles.get(result.status, (result.status.value, "white"))

    # Verdict Header
    console.print()
    verdict_text = (
        f"[bold]Status:[/bold] [{status_color}]{status_label}[/{status_color}]    "
        f"[bold]Latency:[/bold] [yellow]{result.execution_time_ms:.1f} ms[/yellow]    "
        f"[bold]Source:[/bold] [cyan]{result.source}[/cyan]    "
        f"[bold]Overall Confidence:[/bold] [{status_color}]{result.confidence:.3f}[/{status_color}]"
    )
    if result.reason:
        verdict_text += f"\n[dim]Rationale: {result.reason}[/dim]"

    console.print(Panel(verdict_text, title="[bold blue]JEV / CLEF ROUTING VERDICT[/bold blue]", border_style=status_color))

    # Layer 1: Model Selection (Auto-Tier)
    model_table = Table(title="1. AI Model Auto-Selector (Public Models)", show_header=True, header_style="bold cyan")
    model_table.add_column("Candidate AI Model", style="bold")
    model_table.add_column("Provider", style="magenta")
    model_table.add_column("Probability Distribution", style="white", justify="left")
    model_table.add_column("Specialty & Context", style="dim")

    selected_model_name = result.selected_model or (
        max(result.model_probabilities, key=lambda k: result.model_probabilities[k])
        if result.model_probabilities else "gemini_2_0_flash"
    )

    if result.model_probabilities:
        for m_name, prob in sorted(result.model_probabilities.items(), key=lambda x: x[1], reverse=True):
            tier_def = registry.get_model_tier(m_name)
            provider = tier_def.provider if tier_def else "-"
            specialty = f"{tier_def.specialty} ({tier_def.context_window})" if tier_def else ""
            is_selected = (m_name == selected_model_name)
            name_label = f"[bold green]>> {m_name}[/bold green]" if is_selected else f"   {m_name}"
            model_table.add_row(name_label, provider, _render_prob_bar(prob), specialty)
    else:
        tier_def = registry.get_model_tier(selected_model_name)
        provider = tier_def.provider if tier_def else "-"
        specialty = f"{tier_def.specialty} ({tier_def.context_window})" if tier_def else ""
        model_table.add_row(f"[bold green]>> {selected_model_name}[/bold green]", provider, _render_prob_bar(result.confidence), specialty)

    console.print(model_table)

    # Layer 2: Skill & Tool Selection
    action_table = Table(title="2. Specialized Skill & Tool Decision", show_header=True, header_style="bold magenta")
    action_table.add_column("Action / Tool Candidate", style="bold")
    action_table.add_column("Probability Distribution", style="white", justify="left")
    action_table.add_column("Matching Criteria / Role", style="dim")

    if result.probabilities:
        for act, prob in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
            tool_def = registry.get_tool(act)
            criteria_desc = tool_def.criteria if tool_def else ""
            is_selected = (act == result.action)
            action_label = f"[bold green]>> {act}[/bold green]" if is_selected else f"   {act}"
            action_table.add_row(action_label, _render_prob_bar(prob), criteria_desc)
    else:
        action_table.add_row(f"[bold green]>> {result.action}[/bold green]", _render_prob_bar(result.confidence), "")

    if result.selected_skill:
        skill_def = registry.get_skill(result.selected_skill)
        skill_desc = f" ({skill_def.description})" if skill_def else ""
        console.print(f"[bold magenta]Domain Skill Activated:[/bold magenta] [bold white]{result.selected_skill}[/bold white]{skill_desc}")
    console.print(action_table)

    # Layer 3: Safety & Policy Gates
    gate_table = Table(title="3. Safety & Policy Gates (Noul Boolean Evaluation)", show_header=True, header_style="bold yellow")
    gate_table.add_column("Guard Check", style="bold white")
    gate_table.add_column("Result", style="white")
    gate_table.add_column("Calibrated Confidence", style="cyan")
    gate_table.add_column("Policy Enforcement", style="dim")

    dest_label = "[bold red]YES (Destructive)[/bold red]" if result.is_destructive else "[bold green]NO (Non-destructive)[/bold green]"
    conf_label = "[bold yellow]REQUIRED (Pause)[/bold yellow]" if result.requires_confirmation else "[bold green]NOT REQUIRED (Auto-Execute)[/bold green]"
    comp_label = "[bold magenta]COMPLETE (Stop)[/bold magenta]" if result.task_completion else "[bold cyan]IN PROGRESS[/bold cyan]"

    gate_table.add_row(
        "Destructive Action?",
        dest_label,
        f"{result.destructive_confidence:.3f}",
        "Requires sandboxing or confirmation if true",
    )
    gate_table.add_row(
        "Human Confirmation?",
        conf_label,
        "-",
        "Mandatory for destructive terminal or deletion actions",
    )
    gate_table.add_row(
        "Task Completed?",
        comp_label,
        f"{result.task_completion_confidence:.3f}",
        "Terminates agent tool iteration loop when satisfied",
    )
    console.print(gate_table)

    # Layer 4: Downstream Execution Directive
    directive = result.execution_directive or {
        "dispatch_model": result.selected_model or "gemini_2_0_flash",
        "active_skill": result.selected_skill or "general",
        "target_action": result.action,
        "requires_confirmation": result.requires_confirmation,
    }
    dir_table = Table(title="4. Downstream Agent Directive", show_header=True, header_style="bold cyan")
    dir_table.add_column("Directive Parameter", style="bold")
    dir_table.add_column("Configuration Value", style="green")

    for k, v in directive.items():
        dir_table.add_row(str(k), str(v))
    console.print(dir_table)
    console.print()


@main.command()
def models() -> None:
    """List registered publicly available AI models and their routing criteria."""
    registry = create_default_coding_registry()
    table = Table(title="Public AI Models & Routing Criteria", show_header=True)
    table.add_column("Model Name", style="bold cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Specialty", style="yellow")
    table.add_column("Context", style="blue")
    table.add_column("Routing Criteria", style="green")

    for tier in registry.list_model_tiers():
        table.add_row(
            tier.name,
            tier.provider,
            tier.specialty,
            tier.context_window,
            tier.criteria,
        )
    console.print(table)


@main.command()
def skills() -> None:
    """List registered high-level skills and their member tools."""
    registry = create_default_coding_registry()
    table = Table(title="Registered Skills & Domain Criteria", show_header=True)
    table.add_column("Skill Name", style="bold magenta")
    table.add_column("Description", style="white")
    table.add_column("Included Tools", style="cyan")

    for skill in registry.list_skills():
        table.add_row(skill.name, skill.description, ", ".join(skill.tools) or "-")
    console.print(table)


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
