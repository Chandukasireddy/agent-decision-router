"""FastAPI REST service for agent-decision-router."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_decision_router.config import get_settings
from agent_decision_router.models import (
    DecisionResult,
    RouteRequest,
    RouteResponse,
    ToolDefinition,
)
from agent_decision_router.engine import DecisionEngine
from agent_decision_router.registry import create_default_coding_registry
from agent_decision_router.backend import get_backend


# Shared engine instance for FastAPI service
_engine: Optional[DecisionEngine] = None


def get_api_engine() -> DecisionEngine:
    """Retrieve shared API DecisionEngine instance."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = DecisionEngine(
            registry=create_default_coding_registry(),
            backend=get_backend(),
            settings=settings,
        )
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown."""
    # Initialize engine on startup
    get_api_engine()
    yield
    # Cleanup if needed on shutdown


app = FastAPI(
    title="agent-decision-router API",
    description="Confidence-aware tool & skill decision layer for AI coding agents.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for web and client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Healthcheck endpoint reporting backend status and configuration."""
    settings = get_settings()
    engine = get_api_engine()
    return {
        "status": "healthy",
        "backend": settings.router_backend,
        "cloudflare_configured": settings.is_cloudflare_configured,
        "fast_path_cache_enabled": settings.fast_path_cache_enabled,
        "registered_tools_count": len(engine.registry.list_tools()),
    }


@app.post("/v1/route", response_model=RouteResponse, tags=["Routing"])
async def route_action(request: RouteRequest) -> RouteResponse:
    """Route the next tool action given agent state and context.

    Executes through:
    1. Fast-Path Cache check
    2. System 1 Decision model (@cf/cloudflare/clef-flash or local Ollama)
    3. Confidence Gating (>=0.70 executable, 0.40-0.70 lookahead, <0.40 decline)
    """
    engine = get_api_engine()
    try:
        decision = await engine.route(request)
        return RouteResponse(
            success=True,
            decision=decision,
            meta={"backend": engine.settings.router_backend},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing error: {str(exc)}",
        )


@app.post("/v1/register", tags=["Registry"])
async def register_tool(tool: ToolDefinition) -> Dict[str, Any]:
    """Dynamically register a new tool or update an existing one."""
    engine = get_api_engine()
    registered = engine.registry.register_tool(
        name=tool.name,
        description=tool.description,
        criteria=tool.criteria,
        parameters=tool.parameters,
        is_destructive=tool.is_destructive,
        metadata=tool.metadata,
    )
    return {
        "success": True,
        "registered_tool": registered.model_dump(),
        "total_tools": len(engine.registry.list_tools()),
    }


@app.get("/v1/tools", response_model=List[ToolDefinition], tags=["Registry"])
async def list_tools() -> List[ToolDefinition]:
    """Retrieve all registered tools and their routing criteria."""
    engine = get_api_engine()
    return engine.registry.list_tools()


@app.delete("/v1/tools/{tool_name}", tags=["Registry"])
async def unregister_tool(tool_name: str) -> Dict[str, Any]:
    """Unregister a tool from the router."""
    engine = get_api_engine()
    removed = engine.registry.unregister(tool_name)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )
    return {"success": True, "unregistered": tool_name}


@app.get("/v1/cache/stats", tags=["Cache"])
async def cache_stats() -> Dict[str, Any]:
    """Get Stage 1 Fast-Path Cache telemetry and performance statistics."""
    engine = get_api_engine()
    stats = engine.cache.stats.to_dict()
    stats["current_cache_size"] = engine.cache.size()
    return stats


@app.post("/v1/cache/clear", tags=["Cache"])
async def cache_clear() -> Dict[str, Any]:
    """Clear all cached decisions from the Fast-Path Cache."""
    engine = get_api_engine()
    engine.cache.clear()
    return {
        "success": True,
        "message": "Cache successfully cleared",
        "current_cache_size": engine.cache.size(),
    }
