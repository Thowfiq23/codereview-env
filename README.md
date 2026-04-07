---
title: CodeReview-Env V4
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - openenv
  - swe-bench
  - code-review
  - reinforcement-learning
  - execution-sandbox
---

# CodeReview-Env V4

> **An OpenEnv-compliant interactive software engineering benchmark.**
> An LLM agent is dropped into a real Linux sandbox with a failing CI/CD pipeline — it must investigate, patch, and verify real Python bugs. A real `pytest` runner grades every fix.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.dev)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/Thowfiq23/codereview-env-staging)
[![openenv validate](https://img.shields.io/badge/openenv%20validate-6%2F6%20PASS-brightgreen)](#spec-compliance)
[![Docker](https://img.shields.io/badge/Docker-builds%20%26%20runs-blue)](#run-with-docker)

---

## What Is This?

CodeReview-Env puts an agent inside the same loop that every software engineer lives in every day:

```
read failing tests  →  identify root cause  →  patch source  →  verify fix  →  submit
```

This is not a toy. Every action the agent takes — running `pytest`, reading files, writing patches — executes as real shell commands inside a physical Linux sandbox. The grader is `pytest` itself: the agent either fixes the bug or it does not.

---

## Why This Domain

**The problem is real.** Every software organization runs CI/CD pipelines that block merges when tests fail. Engineers spend hours per week triaging test failures, tracing them to root causes, and writing fixes — work that is repetitive, well-specified, and ripe for automation.

**Existing benchmarks fall short for RL.**

| Benchmark | Interactive? | Dense Rewards? | Containerized? | Self-contained? |
|-----------|:---:|:---:|:---:|:---:|
| SWE-bench | No | No — pass/fail only | No | No — needs full repo clone |
| HumanEval | No | No | No | Yes |
| **CodeReview-Env** | **Yes** | **Yes — per patch** | **Yes** | **Yes** |

**Why this fills a gap.** SWE-bench is the gold standard for measuring whether models can fix real bugs, but it is not designed for RL training — it has no intermediate signal, no episode boundary, and no standard API. CodeReview-Env produces a dense reward at every `patch_file` call, giving an RL training loop gradient information throughout the trajectory. An agent that fixes half the tests earns half the credit; one that fixes all tests earns 90% and must still call `submit_review` to confirm, creating an additional incentive layer.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Agent  (inference.py)                  │
│   ReAct loop: observe → think → act → observe → …         │
└────────────────────┬───────────────────────────────────────┘
                     │  HTTP/JSON  (OpenEnv API)
                     ▼
┌────────────────────────────────────────────────────────────┐
│              FastAPI Server  (server/app.py)  :7860        │
│  POST /reset   POST /step   GET /state                     │
│  GET /health   GET /metadata   GET /schema   POST /mcp     │
│                  asyncio.Lock — serialized episode state   │
└────────────────────┬───────────────────────────────────────┘
                     │  subprocess.run()  +  open()
                     ▼
┌────────────────────────────────────────────────────────────┐
│       Physical Sandbox  /tmp/codereview_workspaces/        │
│       /{episode-uuid}/                                     │
│         ├── auth/models.py      ← buggy source, writable  │
│         ├── auth/crypto.py      ← buggy source, writable  │
│         └── tests/test_auth.py  ← read-only oracle        │
└────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Real disk I/O.** `patch_file` calls `open(..., "w")` on the actual filesystem. There is no mocking layer.
- **Real test execution.** `submit_review` calls `pytest tests/ -v` via `subprocess.run()`. Pass rate is parsed from pytest's own output.
- **Isolated episodes.** Each `reset()` creates a new UUID workspace. No state bleeds between episodes.
- **Credential isolation.** Subprocesses receive a minimal environment (`PATH`, `HOME`, `LANG`) — no `HF_TOKEN`, no `OPENAI_API_KEY` can be exfiltrated.
- **Process group isolation.** `start_new_session=True` prevents rogue `kill -9 -1` commands from reaching the FastAPI process.

---

## OpenEnv Spec Compliance

`openenv validate .` → **PASS**
`openenv validate --url` → **6/6 criteria PASS**

| Requirement | Implementation |
|---|---|
| Typed `Action` Pydantic model | `AgentAction` — [models.py](models.py) |
| Typed `Observation` Pydantic model | `CodeObservation` — [models.py](models.py) |
| Typed `Reward` Pydantic model | `EpisodeReward` — [models.py](models.py) |
| Typed `State` Pydantic model | `ReviewState` — [models.py](models.py) |
| `POST /reset` | Returns clean `CodeObservation`, deploys fresh sandbox |
| `POST /step` | Returns `observation`, `reward`, `done`, `info`, `typed_reward` |
| `GET /state` | Returns current `ReviewState` |
| `GET /health` | Returns `{"status": "healthy"}` |
| `GET /metadata` | Returns name, description, version, author |
| `GET /schema` | Returns JSON schemas for action, observation, state |
| `POST /mcp` | JSON-RPC 2.0 — `tools/list` method supported |
| `openenv.yaml` manifest | [openenv.yaml](openenv.yaml) |
| `reward_range: [0.0, 1.0]` | Enforced in grader — never violated |
| `max_steps: 10` | Enforced server-side, episode auto-terminates |
| Dockerfile | [Dockerfile](Dockerfile) — `docker build && docker run` works |
| HF Space | Live, containerized, responds to all endpoints |

---

## Action Space

The agent has exactly three tools. Each turn it outputs one JSON object.

### `execute_command` — Bash Terminal

Run any shell command in the isolated sandbox. Free to call as many times as needed — no reward, no penalty.

```json
{"action_type": "execute_command", "command": "pytest tests/ -v"}
{"action_type": "execute_command", "command": "cat auth/models.py"}
{"action_type": "execute_command", "command": "grep -n 'def hash_password' auth/crypto.py"}
```

**Constraints:** 15-second timeout per command. Output truncated to 4,000 characters (last N chars retained — contains the actionable failure lines).

### `patch_file` — Write a Fix to Disk

Overwrite any source file with the agent's proposed fix. Immediately triggers partial reward grading.

```json
{
  "action_type": "patch_file",
  "target_file": "auth/crypto.py",
  "new_content": "import hashlib\nimport secrets\n\ndef hash_password(password: str):\n    salt = secrets.token_bytes(16)\n    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex() + salt.hex()"
}
```

**Constraints:** Path traversal (`../`) blocked. Writing to `tests/` blocked — the test oracle is read-only. Invalid Python syntax is rejected before touching disk (py_compile check).

### `submit_review` — Final Grader

Declare the fix complete. The environment runs `pytest tests/ -v` one final time against the agent's modified files and assigns the terminal reward. The episode ends regardless of score.

```json
{"action_type": "submit_review", "summary": "Replaced MD5 with PBKDF2 and parameterized the SQL query to eliminate injection."}
```

**Schema:**

```python
class AgentAction(BaseModel):
    action_type: Literal["execute_command", "patch_file", "submit_review"]
    command:     Optional[str]   # Used by execute_command
    target_file: Optional[str]   # Used by patch_file
    new_content: Optional[str]   # Used by patch_file
    summary:     Optional[str]   # Used by submit_review
```

---

## Observation Space

Every `reset()` and `step()` returns a `CodeObservation`:

```python
class CodeObservation(BaseModel):
    task_id:         str        # "task_1_pr" | "task_2_pr" | "task_3_pr"
    context:         str        # Full PR description — the agent's briefing
    available_files: List[str]  # Files present at episode start
    action_result:   str        # stdout/stderr of last command, or status message
    step_number:     int        # Steps consumed so far (1-indexed)
    done:            bool       # True when episode has ended
    reward:          float      # Reward earned this step [0.0, 1.0]
```

The `context` field is the agent's only briefing — it describes the PR, the failing CI checks, and the files to investigate. It does **not** reveal the bugs directly; the agent must read the files to understand what is wrong.

---

## Reward Function

The reward is **dense by design** — every `patch_file` produces a signal proportional to how many tests now pass. This gives RL algorithms gradient information throughout the trajectory, not just at episode end.

```
0.0 ─────────── 0.45 ────────────── 0.90 ──────── 1.0
  execute cmds     1 of 2 bugs fixed   both fixed    submit
```

| Action | Reward | Formula |
|--------|--------|---------|
| `execute_command` | `0.0` | Exploration is free |
| `patch_file` — no tests gained | `0.0` | Bad patch, no signal |
| `patch_file` — partial fix | `0.0 – 0.9` | `(tests_passing / total_tests) × 0.9` |
| `submit_review` — all tests pass | `1.0` | Full credit, terminal |
| `submit_review` — partial tests pass | `0.0 – 1.0` | `tests_passing / total_tests` |
| Max steps (10) exceeded | `0.0` | Episode terminates |

**The 0.9 cap on `patch_file`** creates an intentional final incentive: a perfect patch earns 0.9, but the agent must call `submit_review` to earn the full 1.0. This penalizes "patch and stall" strategies and rewards agents that understand when they are done.

**Episode score** = `max(rewards)` over the trajectory — the highest reward the agent achieved. A policy that solves the task in 5 steps scores the same as one that solves it in 9 steps, but the step count is visible in the trajectory for human analysis.

**Built-in penalties:**
- Attempting path traversal returns an error observation (0.0 reward, no disk write)
- Attempting to overwrite test files returns an error (0.0 reward)
- Syntax-invalid patches are rejected before touching disk (0.0 reward)
- Commands exceeding 15 seconds are killed (0.0 reward, timeout message)

---

## Tasks

### Task 1 — `task_1_pr` — Security Audit — Easy

**Scenario:** *PR #42 "Add user authentication"* — CI/CD security scanner is blocking the merge due to two critical vulnerabilities introduced in the new auth module.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `auth/models.py` | `get_user_query` builds SQL via f-string: `f"SELECT * FROM users WHERE username = '{username}'"` — classic SQL injection | Return a parameterized query: `('SELECT * FROM users WHERE username = %s', (username,))` |
| `auth/crypto.py` | `hash_password` uses `hashlib.md5` — cryptographically broken since 2004 | Replace with SHA-256, bcrypt, or PBKDF2 |

**Grader:** Tests check structural properties (parameterized tuple returned, `md5` not present in hash function) — not runtime values. The agent must understand what a secure pattern looks like, not just copy a string.

**Baseline (Llama-3.3-70B, temp=0.0):** Solved in **7 steps**, score **1.00**
Reward trajectory: `0.00 → 0.00 → 0.45 → 0.00 → 0.90 → 0.00 → 1.00`

---

### Task 2 — `task_2_pr` — Logic Bug — Medium

**Scenario:** *PR #88 "Implement shopping cart billing"* — QA reports checkout totals are wrong. A cart with two $10 items and one $20 item should total $32 after a 20% discount; the code returns $8.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `billing/cart.py` | `calculate_total` sums `item['price']` ignoring `item['quantity']` — off by a factor of quantity | Multiply: `item['price'] * item['quantity']` |
| `billing/discounts.py` | `apply_discount` returns `amount * 0.20` — charges 20% of price instead of keeping 80% | Return `amount * 0.80` |

**Grader:** Tests run the buggy function with known inputs and assert exact numeric output. The agent cannot guess — it must trace the arithmetic.

**Baseline (Llama-3.3-70B, temp=0.0):** Solved in **7 steps**, score **1.00**
Reward trajectory: `0.00 → 0.00 → 0.00 → 0.45 → 0.90 → 0.00 → 1.00`

---

### Task 3 — `task_3_pr` — Security + Async Performance — Hard

**Scenario:** *PR #105 "Integrate Stripe payment processor"* — deployment is blocked by two independent issues: a secrets scanner flagging a hardcoded API key, and a performance regression that blocks the async event loop.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `payments/config.py` | `STRIPE_LIVE_KEY = 'sk_live_9876543210qwerty'` hardcoded | Move to `os.getenv('STRIPE_LIVE_KEY')` |
| `payments/processor.py` | `time.sleep(5)` called synchronously inside a payment handler — blocks the entire event loop under async load | Declare `async def process_payment` and replace with `await asyncio.sleep(5)` |

**Why this is hard:** These two bugs require completely different domain knowledge — secrets management and Python's async/await model — applied simultaneously. The grader for the performance bug uses **AST analysis** (not string matching) to verify that `process_payment` is declared as `async def` and contains an `await` expression. A comment saying `# async def` or a docstring trick will not pass.

**Grader details:**
```python
# Confirms process_payment is truly async — string matching is not sufficient
async_funcs = [n for n in ast.walk(tree)
               if isinstance(n, ast.AsyncFunctionDef) and n.name == 'process_payment']
has_await = any(isinstance(n, ast.Await) for n in ast.walk(async_funcs[0]))
```

**Baseline (Llama-3.3-70B, temp=0.0):** Solved in **8 steps**, score **1.00**
Reward trajectory: `0.00 → 0.00 → 0.45 → 0.00 → 0.00 → 0.90 → 0.00 → 1.00`
*(Step 5 patches processor.py as `def` — grader rejects it. Step 6 corrects to `async def` + `await`.)*

---

## Difficulty Progression

| Task | Domain | Bugs | Steps | Trajectory | Score |
|------|--------|------|-------|------------|-------|
| `task_1_pr` | Security | 2 | 7 | `0→0.45→0.90→1.0` | **1.00** |
| `task_2_pr` | Arithmetic Logic | 2 | 7 | `0→0.45→0.90→1.0` | **1.00** |
| `task_3_pr` | Security + Async | 2 | 8 (+1 failed patch) | `0→0.45→0→0.90→1.0` | **1.00** |

The difficulty is visible in **two places**:
1. **Step count** — task_3 requires more attempts to reach 0.90
2. **Failed patch mid-trajectory** — task_3 produces a `0.00` mid-episode after a syntactically valid but semantically wrong patch (`def` instead of `async def`), which a weaker agent would not self-correct

A policy with random or shallow reasoning will plateau at 0.45 (one bug fixed) or 0.0 (no bugs fixed). The dense gradient gives an RL agent the signal to improve.

---

## Baseline Results

**Model:** `llama-3.3-70b-versatile` via Groq API
**Temperature:** `0.0` (fully deterministic — results are exactly reproducible)
**Infrastructure:** 2 vCPU / 4 GB RAM (well within the 8 GB limit)
**Runtime:** ~3 minutes for all 3 tasks

```
[START] task=task_3_pr env=codereview-env model=llama-3.3-70b-versatile
[STEP] step=1 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.00 done=false error=null
[STEP] step=2 action={"action_type":"execute_command","command":"cat payments/config.py"} reward=0.00 done=false error=null
[STEP] step=3 action={"action_type":"patch_file","target_file":"payments/config.py","new_content":"import os\n\nSTRIPE_TEST_KEY = os.getenv('STRIPE_TEST_KEY')\nSTRIPE_LIVE_KEY = os.getenv('STRIPE_LIVE_KEY')"} reward=0.45 done=false error=null
[STEP] step=4 action={"action_type":"execute_command","command":"cat payments/processor.py"} reward=0.00 done=false error=null
[STEP] step=5 action={"action_type":"patch_file","target_file":"payments/processor.py","new_content":"import asyncio\n\ndef process_payment(user_id, amount):\n    await asyncio.sleep(5)\n    return True"} reward=0.00 done=false error=null
[STEP] step=6 action={"action_type":"patch_file","target_file":"payments/processor.py","new_content":"import asyncio\n\nasync def process_payment(user_id, amount):\n    await asyncio.sleep(5)\n    return True"} reward=0.90 done=false error=null
[STEP] step=7 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.00 done=false error=null
[STEP] step=8 action={"action_type":"submit_review","summary":"Fixed hardcoded Stripe API key and declared process_payment as async with await asyncio.sleep."} reward=1.00 done=true error=null
[END] success=true steps=8 score=1.00 rewards=0.00,0.00,0.45,0.00,0.00,0.90,0.00,1.00

[START] task=task_1_pr env=codereview-env model=llama-3.3-70b-versatile
[STEP] step=1 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.00 done=false error=null
[STEP] step=2 action={"action_type":"execute_command","command":"cat auth/models.py"} reward=0.00 done=false error=null
[STEP] step=3 action={"action_type":"patch_file","target_file":"auth/models.py","new_content":"def get_user_query(username):\n    return ('SELECT * FROM users WHERE username = %s', (username,))"} reward=0.45 done=false error=null
[STEP] step=4 action={"action_type":"execute_command","command":"cat auth/crypto.py"} reward=0.00 done=false error=null
[STEP] step=5 action={"action_type":"patch_file","target_file":"auth/crypto.py","new_content":"import hashlib\nimport secrets\n\ndef hash_password(password: str):\n    salt = secrets.token_bytes(16)\n    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex() + salt.hex()"} reward=0.90 done=false error=null
[STEP] step=6 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.00 done=false error=null
[STEP] step=7 action={"action_type":"submit_review","summary":"Fixed SQL injection with parameterized query and replaced MD5 with PBKDF2."} reward=1.00 done=true error=null
[END] success=true steps=7 score=1.00 rewards=0.00,0.00,0.45,0.00,0.90,0.00,1.00

[START] task=task_2_pr env=codereview-env model=llama-3.3-70b-versatile
[STEP] step=1 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.00 done=false error=null
[STEP] step=2 action={"action_type":"execute_command","command":"cat billing/cart.py"} reward=0.00 done=false error=null
[STEP] step=3 action={"action_type":"execute_command","command":"cat billing/discounts.py"} reward=0.00 done=false error=null
[STEP] step=4 action={"action_type":"patch_file","target_file":"billing/discounts.py","new_content":"def apply_discount(amount):\n    return amount * 0.80"} reward=0.45 done=false error=null
[STEP] step=5 action={"action_type":"patch_file","target_file":"billing/cart.py","new_content":"from billing.discounts import apply_discount\n\ndef calculate_total(items):\n    total = sum(item['price'] * item['quantity'] for item in items)\n    return apply_discount(total)"} reward=0.90 done=false error=null
[STEP] step=6 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.00 done=false error=null
[STEP] step=7 action={"action_type":"submit_review","summary":"Fixed quantity multiplication and discount percentage."} reward=1.00 done=true error=null
[END] success=true steps=7 score=1.00 rewards=0.00,0.00,0.00,0.45,0.90,0.00,1.00

[SUMMARY] tasks=3 scores=1.00,1.00,1.00 average=1.00
```

> Note: The dense trajectory is the key output for RL training. Step 5 of task_3 shows the grader
> correctly rejecting a syntactically valid but semantically wrong patch (`def` without `async`),
> returning `reward=0.00`. The agent self-corrects on step 6 and earns `0.90`. This demonstrates
> that the grader enforces semantic correctness, not just syntactic validity.

---

## Setup and Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized run)
- A Groq / HuggingFace / OpenAI-compatible API key

### Run with Docker

```bash
git clone https://github.com/Thowfiq23/codereview-env
cd codereview-env

docker build -t codereview-env .
docker run -p 7860:7860 codereview-env
```

The server starts on port 7860. Verify with:

```bash
curl http://localhost:7860/health
# {"status": "healthy"}
```

### Run Locally (without Docker)

```bash
pip install -r requirements.txt
PYTHONPATH=$PWD python3 -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run the Baseline Agent

```bash
export HF_TOKEN=your_groq_or_hf_api_key
export MODEL_NAME=llama-3.3-70b-versatile
export API_BASE_URL=https://api.groq.com/openai/v1   # or https://router.huggingface.co/v1
export ENV_URL=http://localhost:7860                  # or the HF Space URL

python3 inference.py
```

### Run Against the Live HF Space

```bash
export ENV_URL=https://thowfiq23-codereview-env-staging.hf.space
python3 inference.py
```

### API Quick Reference

```bash
# Liveness check
curl http://localhost:7860/health

# Environment metadata
curl http://localhost:7860/metadata

# JSON schemas (action / observation / state)
curl http://localhost:7860/schema

# Start a new episode (cycles through tasks: 1 → 2 → 3 → 1 → …)
curl -X POST http://localhost:7860/reset

# Execute a bash command
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "execute_command", "command": "pytest tests/ -v"}'

# Write a file fix (triggers partial reward)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "patch_file", "target_file": "auth/crypto.py", "new_content": "..."}'

# Submit and grade (terminal — episode ends)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_review", "summary": "Fixed SQL injection and MD5."}'

# Inspect current episode state
curl http://localhost:7860/state

# MCP tool listing (JSON-RPC 2.0)
curl -X POST http://localhost:7860/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

---

## Environment State

```python
class ReviewState(BaseModel):
    episode_id:          str    # UUID — unique per reset()
    task_id:             str    # Which task is active
    step_count:          int    # Steps consumed
    current_task_index:  int    # Position in task cycle (0/1/2)
    total_reward:        float  # Best reward achieved so far this episode
    done:                bool   # True when episode has ended
```

`state()` is read-only — it does not advance the episode. Safe to call at any frequency for monitoring.

---

## Security Properties

The environment is designed to be safe to expose over HTTP without a firewall:

| Threat | Defence |
|--------|---------|
| **Path traversal** via `patch_file` (`../../etc/passwd`) | `os.path.realpath()` boundary check — any path outside the workspace is rejected before I/O |
| **Test oracle tampering** (`tests/` overwrite for free 1.0) | `tests/` directory is `chmod 444` at setup + explicitly blocked by name |
| **Syntax bomb** (invalid Python to crash grader) | `py_compile.compile()` runs before any disk write — invalid code returns error observation |
| **Hanging subprocess** (`sleep 1000`, `dd if=/dev/zero`) | 15-second timeout via `subprocess.run(timeout=15)` — `TimeoutExpired` caught cleanly |
| **Process escape** (`kill -9 -1`) | `start_new_session=True` — child process in its own session, cannot signal parent |
| **Credential exfiltration** (`env \| curl ...`) | Subprocesses receive a minimal env (`PATH`, `HOME`, `LANG` only) — no API keys present |
| **Context overflow** (recursive `find /`, `cat /dev/zero`) | Output truncated to 4,000 characters — last N chars retained for actionable content |
| **Unbounded episodes** | Server-side `MAX_STEPS=10` — episode auto-terminates with no reward |
| **Cross-episode contamination** | Fresh UUID workspace per `reset()` — previous workspace deleted on completion |
| **Concurrent state corruption** | `asyncio.Lock` around all `/reset` and `/step` handlers |

---

## Project Structure

```
codereview-env/
├── inference.py           ReAct agent loop — mandatory [START]/[STEP]/[END] stdout format
├── models.py              Pydantic schemas: AgentAction, CodeObservation, EpisodeReward, ReviewState
├── tasks.py               Task definitions — embedded source files, test files, grader metadata
├── client.py              Thin HTTP client for inference.py → server communication
├── openenv.yaml           OpenEnv spec manifest (spec_version, tasks, action/obs models, reward_range)
├── requirements.txt       Python dependencies (fastapi, uvicorn, pydantic, openai, pytest)
├── Dockerfile             Non-root sandbox user, installs deps, starts uvicorn on :7860
├── server/
│   ├── app.py             FastAPI application — all HTTP endpoints, asyncio.Lock, lifespan cleanup
│   └── environment.py     Sandbox engine: workspace management, tool dispatch, pytest grading
└── README.md              This file
```

---

## openenv.yaml

```yaml
spec_version: 1
name: codereview-env
description: >
  SWE-bench style autonomous code review and patching POMDP sandbox.
  Agents fix real bugs in Python repos; pytest grades the fix.
tasks:
  - task_1_pr
  - task_2_pr
  - task_3_pr
action_model: AgentAction
observation_model: CodeObservation
state_model: ReviewState
max_steps: 10
reward_range: [0.0, 1.0]
port: 7860
```

---

## Validation

Run the pre-submission checks before submitting:

```bash
# 1. Structural validation
openenv validate .

# 2. Runtime validation (server must be running)
openenv validate --url http://localhost:7860

# 3. Verify all 3 tasks produce scores in [0.0, 1.0]
python3 inference.py

# 4. Docker build
docker build -t codereview-env .
docker run -p 7860:7860 codereview-env
```

Expected output for `openenv validate --url`:

```json
{"passed": true, "summary": {"passed_count": 6, "total_count": 6, "failed_criteria": []}}
```
