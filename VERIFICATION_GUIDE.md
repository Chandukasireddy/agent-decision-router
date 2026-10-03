# Verification & Testing Guide

This guide provides clear, step-by-step instructions to verify and explore how the **Jev / Clef structured evaluation model** works inside `agent-decision-router`.

---

## 1. What is Jev / Clef in this Project?

Unlike traditional generative LLMs (like GPT-4 or Claude) that output unstructured text token-by-token, **Jev (and Clef)** is a specialized **System 1 evaluation model**:
- **Structured Questions Only**: Evaluates state against typed questions:
  - **`Choice`**: Predicts categorical selections with a calibrated probability distribution across candidates (e.g. Model selection, Tool selection).
  - **`Noul`**: Evaluates calibrated truth values between `0.0` and `1.0` (e.g. `is_destructive`, `task_completion`).
- **Sub-Second Latency**: Runs pre-flight routing in **100–400 ms** before dispatching expensive agent workflows.
- **Calibrated Probabilities**: Gives mathematical confidence scores to prevent hallucinated tool calls and gate unsafe commands.

---

## 2. Quick Setup Check

Before testing, ensure your virtual environment is active:

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\activate

# 2. Verify CLI is recognized
agent-decision-router --help
```

> **Note on Backends:**
> - If `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` are configured in `.env`, the router connects directly to `@cf/cloudflare/clef-flash` via Cloudflare Workers AI.
> - If no token is provided, it seamlessly defaults to the built-in offline `mock` heuristic backend.

---

## 3. Verification Method 1: Interactive Web Dashboard (UI)

The Web Dashboard is the most visual way to explore Jev’s decision-making.

### Step 1: Start the Web Server
```powershell
agent-decision-router serve --port 8000
```

### Step 2: Open the Dashboard
Open your web browser and navigate to:
**[http://localhost:8000](http://localhost:8000)** (or `http://localhost:8000/dashboard`)

### Step 3: Test Scenarios in the Playground Tab
Click the preset scenario chips located directly beneath the prompt box:

| Preset Chip | Prompt Tested | What Jev Evaluates & Selects |
| :--- | :--- | :--- |
| **🧪 Unit Tests** | `Run pytest tests/auth_test.py to inspect login errors` | **Model**: `gemini_2_0_flash` (sub-second iteration speed)<br>**Skill**: `test_and_verification`<br>**Tool**: `run_terminal_command` |
| **🏗️ Architecture Refactor** | `Refactor entire authentication architecture to zero-trust OAuth2 with PKCE` | **Model**: `claude_3_5_sonnet` (SOTA architecture reasoning)<br>**Skill**: `architectural_refactoring` |
| **🧠 Deep Logic** | `Prove race condition in concurrent lock-free queue and verify memory orderings` | **Model**: `deepseek_r1` (deep algorithmic reasoning)<br>**Skill**: `architectural_refactoring` |
| **👁️ UI Vision** | `Inspect screenshot of mobile modal and debug why button overflows` | **Model**: `gpt_4o` (multimodal vision & UI inspection)<br>**Skill**: `codebase_navigation` |
| **✏️ Simple Typo** | `Fix small typo in header comment in README.md` | **Model**: `claude_3_5_haiku` (fast low-cost edit)<br>**Skill**: `architectural_refactoring` |
| **⚠️ Destructive Operation** | `rm -rf /var/lib/docker && drop database production` | **Safety Gate**: Turns **RED** (`Destructive: YES`, `Human Confirmation: REQUIRED`) |

### Step 4: Inspect the 4-Layer Output Stage
After clicking **"Evaluate with Jev / Clef"**, observe how the output is structured:
1. **Layer 0 (Verdict & Latency)**: Displays sub-second execution time (ms), source (`system1`, `cache`, or `lookahead`), and overall status.
2. **Layer 1 (Public AI Models)**: Calibrated probability bars showing how Jev scored each candidate model.
3. **Layer 2 (Skill & Tool Decision)**: Active domain skill and candidate tool ranking.
4. **Layer 3 (Safety & Policy Gates)**: Noul evaluations for destructive commands and task completion status.
5. **Layer 4 (Downstream Directive)**: The exact JSON configuration payload dispatched to the agent runtime.

### Step 5: Explore the Educational Tabs
- **Public AI Models Tab**: View all 6 models with providers, specialties, context windows, and Jev routing criteria.
- **Agent Skills Tab**: View domain skills and their associated tool mappings.
- **How Jev Works Tab**: Visual guide on how Jev's typed questions differ from generative LLMs.

---

## 4. Verification Method 2: Command-Line (CLI)

You can verify all outputs directly from PowerShell.

### 1. Inspect Public AI Models Catalogue
```powershell
agent-decision-router models
```
*Confirms all 6 public models (`claude_3_5_sonnet`, `deepseek_r1`, `gpt_4o`, `gemini_2_0_flash`, `claude_3_5_haiku`, `qwen_2_5_coder_32b`) are registered with their context windows and criteria.*

### 2. Inspect Domain Skills
```powershell
agent-decision-router skills
```
*Lists `test_and_verification`, `architectural_refactoring`, `security_and_auth_audit`, `codebase_navigation`, `system_and_devops`, and `user_consultation`.*

### 3. Route a Test Command
```powershell
agent-decision-router route "User requested: Run pytest tests/auth_test.py to inspect login errors"
```
*Expected Output:*
- **Verdict**: Sub-second latency (`~300-600 ms`)
- **Auto Model**: `gemini_2_0_flash` (~78% probability)
- **Domain Skill**: `test_and_verification`
- **Next Tool**: `run_terminal_command` (~70% probability)
- **Safety Gate**: Human Confirmation `REQUIRED` (due to shell execution)
- **Downstream Directive**: Dispatches to `gemini_2_0_flash` with `APPROVAL_REQUIRED` sandbox policy

### 4. Route a Complex Refactoring Command
```powershell
agent-decision-router route "Refactor auth logic to use PKCE and eliminate timing attack vulnerability"
```
*Expected Output:*
- **Auto Model**: `claude_3_5_sonnet` (~77% probability)
- **Domain Skill**: `architectural_refactoring`

### 5. Test JSON Pipeline Output
For integration into automated agent runtimes, add `--json`:
```powershell
agent-decision-router route "Search codebase for OAuth callback handler" --json
```

---

## 5. Verification Method 3: REST API

If you have the server running (`agent-decision-router serve`), you can verify the endpoints via PowerShell or curl:

### Health Check Endpoint
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/v1/health"
```
*Returns backend status, cache state, and registered model counts.*

### Route Decision Endpoint
```powershell
$body = @{ state = "Run pytest tests/auth_test.py"; skip_cache = $true } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/v1/route" -Method Post -Body $body -ContentType "application/json"
```

---

## 6. Verification Method 4: Automated Test Suite

Run the full pytest suite to verify all 35 tests pass with zero errors:

```powershell
pytest
```

*Expected output:*
```text
tests\test_api.py ....                                                   [ 11%]
tests\test_backends.py ......                                            [ 28%]
tests\test_cache.py .....                                                [ 42%]
tests\test_engine.py .......                                             [ 62%]
tests\test_lookahead.py ..                                               [ 68%]
tests\test_mcp.py ...                                                    [ 77%]
tests\test_model_selection.py ...                                        [ 85%]
tests\test_registry.py .....                                             [100%]

============================= 35 passed in 3.19s =============================
```

---

## 7. Summary of Capabilities Verified

By completing these steps, you have verified:
1. **Model Auto-Selection**: How Jev dynamically selects among public models based on task type.
2. **Skill Activation**: How high-level domain skills are matched to developer prompts.
3. **Calibrated Confidence**: How probability bars give transparency into model certainty.
4. **Safety Gating**: How Noul boolean evaluation blocks destructive commands and flags task completion.
5. **Multi-Surface Output**: High-performance delivery across Web UI, CLI tables, and JSON REST API.
