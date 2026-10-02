# agent-decision-router

A framework-agnostic, confidence-aware tool and skill decision layer for AI coding agents.

`agent-decision-router` provides a structured, high-speed decision pipeline that answers the question: **"What should the agent execute next?"** By utilizing typed criteria questions, zero-latency content-addressable caching, and strict confidence gating, it prevents hallucinated tool calls and provides safe, deterministic control over coding agent trajectories.

---

## Key Features

- **Dynamic Tool & Skill Registry:** Register tools and high-level skills with declarative routing criteria, parameters, and destructive risk flags. Dynamically generates typed decision questions.
- **System 1 Decision Backends:**
  - **Cloudflare Workers AI:** Native client running `@cf/cloudflare/clef-flash` (free tier) with probability distributions over criteria options.
  - **Local Ollama / System One:** Interchangeable local provider connecting to `http://localhost:11434/v1/systemone` (e.g., `nimble`).
  - **Mock / Heuristic Engine:** Offline deterministic & temperature-softmax scoring backend for testing without cloud credentials.
- **3-Stage Routing Pipeline:**
  1. **Stage 1 (Fast-Path Cache):** Content-addressable SHA-256 state hashing with LRU eviction and TTL for sub-millisecond repeated routing.
  2. **Stage 2 (System 1 Decision):** Formats state and registered criteria into typed choice and noul questions (`next_tool`, `is_destructive`, `task_completion`).
  3. **Stage 3 (Confidence Gate):**
     - **$\ge 0.70$:** Immediate execution (`EXECUTABLE`).
     - **$0.40 \le \text{confidence} < 0.70$:** Shallow bounded lookahead (MCTS heuristic evaluating prerequisites, repeat penalties, and trajectory alignment).
     - **$< 0.40$:** Refuses execution (`DECLINED`) with `"reason": "low_confidence"` to prevent hallucinated tool calls.
- **Destructive Guardrails & Task Completion:** Automatically flags mutating actions (file drops, git resets, terminal drops) with `requires_confirmation = True`, and detects terminal task satisfaction (`action: "COMPLETE_TASK"`).
- **PreDecision / PostTool Lifecycle:** Intercept routing before execution and record trajectory telemetry after tool runs.
- **FastMCP Server:** Expose routing as an MCP tool for Claude Code, Cline, Cursor, and Codex.
- **FastAPI REST Service & Rich CLI:** Standalone HTTP microservice and interactive terminal CLI.

---

## Architecture

```
                    ┌────────────────────────┐
                    │      Agent State       │
                    │ (Prompt + Trajectory)  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  PreDecision Lifecycle │
                    └───────────┬────────────┘
                                │
                                ▼
                     [Stage 1: Fast-Path Cache]
                     Exact State SHA-256 Hash
                     ├── Hit ───────────────► Return Cached Decision (0.05ms)
                     └── Miss
                           │
                           ▼
                     [Stage 2: System 1 Model]
                     Cloudflare (@cf/cloudflare/clef-flash)
                     or Local Ollama (/v1/systemone)
                           │
                           ▼
                     [Stage 3: Confidence Gate]
                     ├── Confidence >= 0.70 ──► EXECUTABLE
                     ├── 0.40 <= Conf < 0.70 ──► Bounded MCTS Lookahead
                     └── Confidence < 0.40 ───► DECLINE (Low Confidence)
                                │
                                ▼
                     [Destructive & Task Gates]
                     ├── is_destructive ─────► requires_confirmation: true
                     └── task_completion ────► action: COMPLETE_TASK
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Decision Result     │
                    └────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/your-org/agent-decision-router.git
cd agent-decision-router
pip install -e .
```

### Optional Dependencies

```bash
# For development and tests
pip install -e ".[dev]"
```

---

## Configuration

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

```env
# Cloudflare Workers AI
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
CLOUDFLARE_API_TOKEN=your_api_token_here
CLOUDFLARE_MODEL=@cf/cloudflare/clef-flash

# Local Ollama / System One
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=nimble

# Backend Selection: 'cloudflare', 'ollama', or 'mock'
ROUTER_BACKEND=cloudflare

# Thresholds
CONFIDENCE_THRESHOLD_HIGH=0.70
CONFIDENCE_THRESHOLD_LOW=0.40

# Cache
FAST_PATH_CACHE_ENABLED=true
FAST_PATH_CACHE_TTL_SECONDS=3600
FAST_PATH_CACHE_MAX_SIZE=1024
```

---

## Quickstart

### 1. Python Library Usage

```python
from agent_decision_router import (
    DecisionEngine,
    ToolRegistry,
    RouteRequest,
    create_default_coding_registry,
    get_backend,
)

# 1. Use standard coding agent registry or create your own
registry = create_default_coding_registry()

# 2. Add custom tool with criteria
registry.register_tool(
    name="run_database_migration",
    description="Execute alembic or prisma migration scripts",
    criteria="Run database schema migration, alembic upgrade, or prisma db push",
    is_destructive=True,
)

# 3. Initialize decision engine
engine = DecisionEngine(
    registry=registry,
    backend=get_backend(),  # Loads from .env (Cloudflare, Ollama, or Mock)
)

# 4. Route next action
state = "User requested: Run alembic upgrade head to apply pending migrations."
decision = engine.route_sync(RouteRequest(state=state))

print(f"Action:                {decision.action}")
print(f"Status:                {decision.status.value}")
print(f"Confidence:            {decision.confidence:.2f}")
print(f"Requires Confirmation: {decision.requires_confirmation}")
```

### 2. PreDecision & PostTool Agent Lifecycle

```python
from agent_decision_router import DecisionEngine, AgentState

engine = DecisionEngine()
state = AgentState(user_goal="Fix failing auth tests in test_auth.py")

# Step A: Route next action
decision = engine.route_sync(state)

# Step B: Execute action and record result via lifecycle hook
engine.record_tool_result(
    tool=decision.action,
    arguments={"command": "pytest test_auth.py"},
    output="AssertionError: Expected 200, got 401",
    success=False,
    state=state,
)

# Step C: Router automatically factors recent output into next decision
next_decision = engine.route_sync(state)
```

---

## Model Provider Details

### Cloudflare Workers AI (`@cf/cloudflare/clef-flash`)
The primary production backend sends typed schema questions to Cloudflare Workers AI free tier:

```http
POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/clef-flash
Authorization: Bearer {CLOUDFLARE_API_TOKEN}
Content-Type: application/json
```

Payload format:
```json
{
  "state": "User requested: Run the unit tests and fix any assertion errors in auth_test.py. Current git status: 1 modified file.",
  "questions": {
    "next_tool": {
      "type": "choice",
      "instructions": "Which tool should the agent run next?",
      "criteria": {
        "run_terminal_command": "Execute shell commands, pytest, or build tools",
        "read_code_file": "Inspect contents of a file on disk",
        "apply_code_patch": "Write or modify existing source files",
        "ask_user_clarification": "Task is ambiguous or requires human input"
      }
    },
    "is_destructive": {
      "type": "noul",
      "instructions": "Will this planned action delete data, overwrite uncommitted changes, or terminate processes?"
    },
    "task_completion": {
      "type": "noul",
      "instructions": "Has the user's task or objective been completely satisfied and no further tool execution is needed?"
    }
  }
}
```

### Local Ollama (`/v1/systemone`)
For offline or local development, set `ROUTER_BACKEND=ollama` to connect to `http://localhost:11434/v1/systemone` running `nimble`.

---

## MCP Server Integration

`agent-decision-router` provides a FastMCP server that exposes `route_next_action` directly to AI coding assistants (Claude Code, Cursor, Cline, Codex).

### Running the MCP Server

```bash
agent-decision-router mcp --transport stdio
```

### Claude Desktop / Cline Configuration (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "agent-decision-router": {
      "command": "python",
      "args": ["-m", "agent_decision_router.mcp_server"],
      "env": {
        "CLOUDFLARE_ACCOUNT_ID": "your_account_id",
        "CLOUDFLARE_API_TOKEN": "your_api_token",
        "ROUTER_BACKEND": "cloudflare"
      }
    }
  }
}
```

---

## FastAPI REST Service

Start the HTTP API server:

```bash
agent-decision-router serve --host 0.0.0.0 --port 8000
```

### Endpoints

- `POST /v1/route`: Route next tool action.
- `POST /v1/register`: Register new tool and criteria.
- `GET /v1/tools`: List all registered tools.
- `DELETE /v1/tools/{name}`: Unregister tool.
- `GET /v1/cache/stats`: Telemetry and cache hit rates.
- `POST /v1/cache/clear`: Flush cache.
- `GET /v1/health`: Server and backend healthcheck.

#### Example `curl` Request

```bash
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "state": "User wants to run pytest on tests/test_auth.py to verify login fixes.",
    "skip_cache": false
  }'
```

Response:
```json
{
  "success": true,
  "decision": {
    "action": "run_terminal_command",
    "confidence": 0.88,
    "status": "EXECUTABLE",
    "is_destructive": true,
    "requires_confirmation": true,
    "task_completion": false,
    "source": "system1",
    "execution_time_ms": 12.4
  },
  "meta": {
    "backend": "cloudflare"
  }
}
```

---

## Command Line Interface (CLI)

```bash
# View available commands
agent-decision-router --help

# List registered tools and routing criteria
agent-decision-router tools

# Test route a prompt from the terminal
agent-decision-router route "Inspect models.py to check field types"

# Launch FastMCP server
agent-decision-router mcp

# Start REST API server
agent-decision-router serve --port 8000
```

---

## Running Tests

Run the complete test suite with pytest:

```bash
pytest -v
```

All 32 test cases cover:
- Dynamic tool & skill registry criteria generation
- Stage 1 Fast-Path Cache determinism, LRU eviction, and TTL expiration
- Cloudflare & Ollama request payload formatting and response parsing
- Stage 3 confidence gating ($\ge 0.70$ executable, $0.40-0.70$ lookahead, $<0.40$ decline)
- Bounded shallow lookahead / MCTS heuristic
- FastMCP server tools
- FastAPI endpoints

---

## License

MIT License.
