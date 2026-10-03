"""Maze simulation and System 1 decision engine for agent navigation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# Ensure root agent_decision_router can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_decision_router.config import get_settings
from agent_decision_router.models import (
    DecisionPayload,
    QuestionDefinition,
    QuestionType,
    DecisionResult,
    DecisionStatus,
)
from agent_decision_router.cache import FastPathCache
from agent_decision_router.backend.cloudflare import CloudflareBackend
from agent_decision_router.backend.mock import MockBackend


class MazeCell(BaseModel):
    x: int
    y: int
    is_wall: bool = False
    is_start: bool = False
    is_goal: bool = False


# Direction vectors: (dx, dy)
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "move_north": (0, -1),
    "move_south": (0, 1),
    "move_east": (1, 0),
    "move_west": (-1, 0),
}

DIR_NAMES: Dict[str, str] = {
    "move_north": "North (Up)",
    "move_south": "South (Down)",
    "move_east": "East (Right)",
    "move_west": "West (Left)",
}


# Intricate multi-layer mazes with 2-step straight start, multiple diversions, spikes, and dead ends
MAZE_PRESETS: Dict[str, List[str]] = {
    "catacomb_trial": [
        "S........#.....",  # Row 0: 2 straight steps (0,0)->(1,0)->(2,0), then immediate branches East & South
        "##.#########.##",  # Row 1: thin stone walls & vertical shafts
        ".....#.........",  # Row 2: cross-corridors & dead-end spurs
        "####.#.#.#####.",  # Row 3: dividers
        ".......#.....#.",  # Row 4: multiple path branches
        "####.##########",  # Row 5: tactical barriers
        "...#...........",  # Row 6: transverse route
        "##.#.###.######",  # Row 7: obstacles
        ".....#.........",  # Row 8: mid-sanctum chamber
        "##.###.#.###.##",  # Row 9: spike dividers
        ".....#.#...#.#.",  # Row 10: winding lower layers
        ".#.#.#.###.###.",  # Row 11: funnel passages
        ".#.#.#...#....G",  # Row 12: final sanctum (Goal G at 14, 12)
    ],
    "forked_labyrinth": [
        "S......#.....#.#.",  # Row 0: 2 straight steps, then deep divergent branches
        "##.#######.###.#.",  # Row 1: thin partitions
        ".....#...........",  # Row 2: multi-directional paths
        "####.#.#.#######.",  # Row 3: long barrier
        ".......#.....#.#.",  # Row 4: forks & dead ends
        "####.#########.##",  # Row 5: central divide
        "...#.........#.#.",  # Row 6: lower gallery
        "##.#.###.#####.#.",  # Row 7: spike corridors
        ".....#...........",  # Row 8: wide transverse
        "##.###.#.###.#.#.",  # Row 9: room pillars
        ".....#.#...#.#.#.",  # Row 10: winding approach
        ".#.###.###.###.#.",  # Row 11: final chicane
        ".#.#.....#...#.#G",  # Row 12: Goal G at 16, 12
    ],
    "grand_sanctum": [
        "S........#.#.......",  # Row 0: 19-col expansive labyrinth
        "##.#.#####.#.######",  # Row 1: intricate masonry walls
        "...#.#.............",  # Row 2: labyrinth branches
        "##.#.#.###.#####.##",  # Row 3: barrier network
        ".#.#...#.#.....#...",  # Row 4: multiple dead ends
        ".#.#.###.#######.##",  # Row 5: pillars & chambers
        "...#.......#.#.....",  # Row 6: transverse avenue
        ".#.#.#######.######",  # Row 7: deep divider
        ".#.#.#.....#.#.#.#.",  # Row 8: grid chambers
        "##.#####.###.#.#.#.",  # Row 9: inner defenses
        "...#...............",  # Row 10: wide gallery
        "##.#.###.#######.#.",  # Row 11: approach passages
        ".......#.....#...#G",  # Row 12: Goal G at 18, 12
    ],
}


class MazeEnvironment:
    """Grid maze environment with obstacle detection and distance heuristics."""

    def __init__(self, preset_name: str = "catacomb_trial") -> None:
        self.preset_name = preset_name
        self.load_preset(preset_name)

    def load_preset(self, preset_name: str) -> None:
        grid_lines = MAZE_PRESETS.get(preset_name, MAZE_PRESETS["catacomb_trial"])
        self.preset_name = preset_name
        self.height = len(grid_lines)
        self.width = len(grid_lines[0])
        self.grid: List[List[str]] = [list(row) for row in grid_lines]

        self.start_pos = (0, 0)
        self.goal_pos = (self.width - 1, self.height - 1)

        for y, row in enumerate(self.grid):
            for x, char in enumerate(row):
                if char == "S":
                    self.start_pos = (x, y)
                elif char == "G":
                    self.goal_pos = (x, y)

        self.agent_pos = self.start_pos
        self.path_history: List[Tuple[int, int]] = [self.start_pos]
        self.visited_counts: Dict[Tuple[int, int], int] = {self.start_pos: 1}

    def reset(self) -> None:
        self.agent_pos = self.start_pos
        self.path_history = [self.start_pos]
        self.visited_counts = {self.start_pos: 1}

    def is_valid_tile(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.grid[y][x] != "#"

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def get_valid_moves(self, pos: Optional[Tuple[int, int]] = None) -> Dict[str, Tuple[int, int]]:
        cur_x, cur_y = pos or self.agent_pos
        valid: Dict[str, Tuple[int, int]] = {}

        for dir_name, (dx, dy) in DIRECTIONS.items():
            nx, ny = cur_x + dx, cur_y + dy
            if self.is_valid_tile(nx, ny):
                valid[dir_name] = (nx, ny)

        return valid

    def get_decision_moves(self) -> Dict[str, Tuple[int, int]]:
        """Return valid candidate moves, enforcing anti-reversal forward progress.

        If forward corridor paths exist, immediate 180° backtracking to prev_pos is
        excluded to eliminate 2-cell oscillation. If at a dead end, backtrack is permitted.
        """
        valid = self.get_valid_moves()
        # Enforce rule: never step back to Start (0, 0) from the opening corridor
        if self.agent_pos == (1, 0) and "move_east" in valid:
            return {"move_east": valid["move_east"]}

        if len(self.path_history) > 1:
            prev_pos = self.path_history[-2]
            forward = {k: v for k, v in valid.items() if v != prev_pos}
            if forward:
                return forward
        return valid

    def build_clef_state_and_questions(self) -> Tuple[str, Dict[str, QuestionDefinition]]:
        """Construct the System 1 / Jev prompt and criteria schema for Clef."""
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        dist_to_goal = self.manhattan_distance(self.agent_pos, self.goal_pos)
        all_moves = self.get_valid_moves()
        decision_moves = self.get_decision_moves()

        prev_pos = self.path_history[-2] if len(self.path_history) > 1 else None

        # Build environmental spatial description
        state_parts = [
            f"Agent position: ({ax}, {ay}). Final Goal G: ({gx}, {gy}). Manhattan distance: {dist_to_goal} units.",
            f"Step count so far: {len(self.path_history) - 1}.",
        ]

        if prev_pos:
            state_parts.append(f"Immediately stepped from previous tile: ({prev_pos[0]}, {prev_pos[1]}).")

        # Describe surrounding tiles
        surroundings = []
        for d_key, (dx, dy) in DIRECTIONS.items():
            tx, ty = ax + dx, ay + dy
            if not self.is_valid_tile(tx, ty):
                surroundings.append(f"{DIR_NAMES[d_key]}: WALL/BLOCKED")
            elif (tx, ty) == prev_pos and len(decision_moves) > 0 and d_key not in decision_moves:
                surroundings.append(f"{DIR_NAMES[d_key]}: ({tx}, {ty}) -> IMMEDIATE PREVIOUS TILE (do not backtrack)")
            else:
                target_dist = self.manhattan_distance((tx, ty), self.goal_pos)
                visits = self.visited_counts.get((tx, ty), 0)
                revisit_warn = f", visited {visits}x" if visits > 0 else ", unvisited forward path"
                closer = "closer to goal" if target_dist < dist_to_goal else "corridor route"
                surroundings.append(f"{DIR_NAMES[d_key]}: OPEN ({tx}, {ty}) -> {closer} (dist {target_dist}{revisit_warn})")

        state_parts.append("Sensory scan:\n" + "\n".join(f"- {s}" for s in surroundings))
        state_str = "\n".join(state_parts)

        # Generate choice criteria dynamically from decision_moves
        criteria: Dict[str, str] = {}
        for d_key, (tx, ty) in decision_moves.items():
            target_dist = self.manhattan_distance((tx, ty), self.goal_pos)
            visits = self.visited_counts.get((tx, ty), 0)
            if (tx, ty) == prev_pos:
                bonus = "Dead-end escape: backtrack out of trapped pocket"
            elif visits == 0:
                bonus = "Forward unexplored corridor advancing towards goal G"
            else:
                bonus = f"Previously traversed path ({visits} visits) - prioritize unvisited paths"
            criteria[d_key] = f"Step {DIR_NAMES[d_key]} to tile ({tx}, {ty}). {bonus}."

        if not criteria:
            criteria["backtrack"] = "Backtrack to previous tile"
            criteria["stay"] = "Remain in place (suboptimal - delays mission)"
        elif len(criteria) == 1:
            criteria["stay"] = "Remain in place (suboptimal - delays mission, advance forward instead)"

        questions: Dict[str, QuestionDefinition] = {
            "next_tool": QuestionDefinition(
                type=QuestionType.CHOICE.value,
                instructions="Which forward direction should the agent step to navigate towards Goal G without looping?",
                criteria=criteria,
            ),
            "is_destructive": QuestionDefinition(
                type=QuestionType.NOUL.value,
                instructions="Will this choice enter an inescapable dead end or cause an infinite loop?",
            ),
            "task_completion": QuestionDefinition(
                type=QuestionType.NOUL.value,
                instructions="Has the agent reached the destination tile G?",
            ),
        }

        return state_str, questions


class MazeDecisionController:
    """Manages live navigation decisions using Clef Workers AI with Fast-Path Cache."""

    def __init__(self, environment: MazeEnvironment) -> None:
        self.env = environment
        self.settings = get_settings()
        self.cache = FastPathCache(max_size=512, ttl_seconds=3600)

        # Backend provider
        if self.settings.is_cloudflare_configured:
            self.backend = CloudflareBackend()
        else:
            self.backend = MockBackend(auto_heuristic=True)

    def step(self) -> Dict[str, Any]:
        """Compute and execute one step using Clef System 1 decision model."""
        if self.env.agent_pos == self.env.goal_pos:
            return {
                "action": "COMPLETE_TASK",
                "status": "TASK_COMPLETED",
                "confidence": 1.0,
                "is_goal": True,
                "agent_pos": self.env.agent_pos,
                "path_history": self.env.path_history,
                "reason": "Agent has reached the final goal tile G!",
                "probabilities": {},
                "source": "environment",
                "execution_time_ms": 0.1,
            }

        state_str, questions = self.env.build_clef_state_and_questions()
        criteria_schema = questions["next_tool"].criteria or {}

        # Stage 1: Fast-Path Cache
        state_hash = FastPathCache.compute_state_hash(state_str, criteria_schema)
        cached = self.cache.get(state_hash)

        if cached is not None:
            move_action = cached.action
            execution_time = 0.05
            source = "cache"
            status = "CACHE_HIT"
            confidence = cached.confidence
            probabilities = cached.probabilities
            is_destructive = cached.is_destructive
        else:
            # Stage 2: Live Cloudflare Clef-Flash inference
            payload = DecisionPayload(state=state_str, questions=questions)
            parsed = self.backend.query_decision_sync(payload)

            move_action = parsed.selected_action
            confidence = parsed.confidence
            probabilities = parsed.probabilities
            is_destructive = parsed.is_destructive
            source = "cloudflare_clef" if self.settings.is_cloudflare_configured else "mock_engine"
            status = "EXECUTABLE" if confidence >= 0.70 else "LOOKAHEAD_RESOLVED"
            execution_time = 450.0  # Estimated or measured

            # Store in cache
            self.cache.set(
                state_hash,
                DecisionResult(
                    action=move_action,
                    confidence=confidence,
                    status=DecisionStatus.EXECUTABLE if confidence >= 0.70 else DecisionStatus.LOOKAHEAD_RESOLVED,
                    probabilities=probabilities,
                    is_destructive=is_destructive,
                )
            )

        # Apply move to environment with anti-oscillation constraint
        decision_moves = self.env.get_decision_moves()
        valid_moves = self.env.get_valid_moves()
        executed_move = move_action

        if move_action in decision_moves:
            next_pos = decision_moves[move_action]
        elif decision_moves:
            # Pick valid forward candidate if model proposed out-of-schema action
            executed_move = list(decision_moves.keys())[0]
            next_pos = decision_moves[executed_move]
        elif move_action in valid_moves:
            next_pos = valid_moves[move_action]
        else:
            next_pos = self.env.agent_pos

        self.env.agent_pos = next_pos
        self.env.path_history.append(next_pos)
        self.env.visited_counts[next_pos] = self.env.visited_counts.get(next_pos, 0) + 1

        is_goal = (self.env.agent_pos == self.env.goal_pos)

        return {
            "action": executed_move,
            "status": "TASK_COMPLETED" if is_goal else status,
            "confidence": round(confidence, 3),
            "is_goal": is_goal,
            "agent_pos": self.env.agent_pos,
            "path_history": self.env.path_history,
            "probabilities": probabilities,
            "is_destructive": is_destructive,
            "source": source,
            "state_prompt": state_str,
            "execution_time_ms": execution_time,
            "cache_stats": self.cache.stats.to_dict(),
        }
