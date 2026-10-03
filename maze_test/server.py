"""FastAPI application serving the Maze Decision Visualizer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from maze_test.maze_engine import MazeEnvironment, MazeDecisionController, MAZE_PRESETS

app = FastAPI(title="Clef System 1 Maze Navigator")

# Initialize global environment and controller
env = MazeEnvironment(preset_name="catacomb_trial")
controller = MazeDecisionController(env)

STATIC_DIR = BASE_DIR / "static"


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    """Retrieve full maze board and agent status."""
    return {
        "preset": env.preset_name,
        "width": env.width,
        "height": env.height,
        "grid": env.grid,
        "start_pos": env.start_pos,
        "goal_pos": env.goal_pos,
        "agent_pos": env.agent_pos,
        "path_history": env.path_history,
        "is_goal": env.agent_pos == env.goal_pos,
        "presets_available": list(MAZE_PRESETS.keys()),
    }


@app.post("/api/step")
def step_agent() -> Dict[str, Any]:
    """Trigger one decision step using Clef / System 1 engine."""
    try:
        step_result = controller.step()
        return step_result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/reset")
def reset_maze() -> Dict[str, Any]:
    """Reset agent back to start position."""
    env.reset()
    return get_state()


class PresetRequest(BaseModel):
    preset: str


@app.post("/api/preset")
def change_preset(req: PresetRequest) -> Dict[str, Any]:
    """Switch to a different maze layout preset."""
    if req.preset not in MAZE_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")
    env.load_preset(req.preset)
    controller.cache.clear()
    return get_state()


# Mount static directory for HTML, CSS, JS
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def main():
    print("Starting Clef Maze Game Visualizer on http://localhost:8080 ...")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
