---
title: CodeReview-Env V6
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
  - incident-repair
---

# CodeReview-Env V6

> **The execution-grounded software incident repair benchmark for RL agents.**
> An LLM agent is dropped into a real Linux sandbox with a failing CI/CD pipeline or a downed service. It runs real shell commands, writes real file patches, restarts real services, and is graded by real `pytest` — no simulation layer, no mocked test results.

**Quick stats**

| | |
|---|---|
| Tasks | 10 across 5 difficulty tiers |
| Action types | 9 (`run_tests`, `read_file`, `search_codebase`, `inspect_logs`, `patch_file`, `restart_service`, `submit_fix`, `submit_review`, `execute_command`) |
| Instance diversity | Seeded parametric factories — unique problem per episode reset |
| Execution model | Real subprocess · real filesystem · real pytest oracle |
| Reward shape | Dense delta reward on every `patch_file` + 6-component terminal reward |
| Difficulty ceiling | Tasks 5–10 not saturated by greedy patching |
| Stateful tasks | Tasks 7–9 require `restart_service` to apply changes; service/migration state persists across steps |
| Causal state graph | `system_health`, `progress_depth`, `trap_count` tracked — premature restarts penalized |
| Diagnosis scoring | `root_cause` keyword-matched against per-task ground truth (15% of terminal reward) |
| Anti-exploit hardening | 13 tests covering conftest injection, chmod bypass, path traversal, reward farming |
| Spec compliance | OpenEnv `openenv validate` — 10/10 PASS |

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.dev)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/Thowfiq23/CodeReview-Env)
[![openenv validate](https://img.shields.io/badge/openenv%20validate-10%2F10%20PASS-brightgreen)](#spec-compliance)
[![Docker](https://img.shields.io/badge/Docker-builds%20%26%20runs-blue)](#run-with-docker)

---

## What Is This?

CodeReview-Env puts an agent inside the same loop that every software engineer lives in every day:

```
read failing tests  →  inspect logs  →  identify root cause  →  patch source  →  verify fix  →  submit
```

This is not a toy. Every action the agent takes — running `pytest`, reading files, writing patches, restarting services — executes as real shell commands inside a physical Linux sandbox. Test quality is graded by `pytest` directly, but the terminal reward is a 6-component weighted score: test quality, root-cause diagnosis, step efficiency, exploration breadth, trap avoidance, and submission credit — the agent either fixes the bug, explains why, and does so efficiently, or it does not.

---

## Why This Domain

**The problem is real.** Every software organization runs CI/CD pipelines that block merges when tests fail. Engineers spend hours per week triaging test failures, tracing them to root causes, and writing fixes — work that is repetitive, well-specified, and ripe for automation.

**Existing benchmarks fall short for RL.**

| Benchmark | Interactive? | Dense Rewards? | Containerized? | Self-contained? |
|-----------|:---:|:---:|:---:|:---:|
| SWE-bench | No | No — pass/fail only | No | No — needs full repo clone |
| HumanEval | No | No | No | Yes |
| **CodeReview-Env** | **Yes** | **Yes — per patch + 6-component terminal** | **Yes** | **Yes** |

**Why this fills a gap.** SWE-bench is the gold standard for measuring whether models can fix real bugs, but it is not designed for RL training — it has no intermediate signal, no episode boundary, and no standard API. CodeReview-Env produces a dense reward at every `patch_file` call, giving an RL training loop gradient information throughout the trajectory. The 6-component terminal reward further signals *how well* the agent fixed the problem: test quality, root-cause accuracy, step efficiency, exploration breadth, trap avoidance, and completion credit.

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
│         ├── app/models.py        ← buggy source, writable  │
│         ├── config/settings.py   ← buggy source, writable  │
│         └── tests/test_*.py      ← read-only oracle        │
└────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Real subprocess execution** — no AST interpreters, no sandboxed Python eval. `pytest` runs as a child process; file I/O is real filesystem I/O. Subprocesses receive a minimal clean environment (`PATH`, `HOME`, `LANG`, `LC_ALL`, `TMPDIR`, `USER`, `LOGNAME`) — no parent secrets, no ambient Python paths.
- **Parametric seeding** — every task factory derives variable values (timeouts, thresholds, cache sizes) from a SHA-256-seeded RNG, so each episode reset produces a structurally identical but numerically distinct problem.
- **Causal state graph** — for service/incident tasks, the environment tracks `system_health` (`unknown` → `crashed` / `oom_killed` → `degraded` → `running` → `healthy`) and penalizes trap actions that worsen state or waste a restart call.
- **Dense + terminal rewards** — `patch_file` emits `0.01 + Δpass_rate × 0.88` immediately; `submit_fix` triggers 6-component terminal scoring.
- **MCP JSON-RPC 2.0** — `/mcp` endpoint exposes all 9 tools via `tools/list` and `tools/call` for MCP-native clients.

---

## Task Suite

### Difficulty Overview

| Task | Incident | Difficulty | Bugs | Key Challenge |
|------|----------|------------|------|---------------|
| INC-001 (`task_1_pr`) | User Registration | medium | 3 | Email regex, password length, MD5→SHA-256 |
| INC-002 (`task_2_pr`) | Order Pricing Engine | medium-hard | 3 | Quantity multiply, discount direction, tax formula |
| INC-003 (`task_3_pr`) | Async Payment Processor | hard | 3 | Hardcoded API key, sync/async mismatch, retry off-by-one |
| INC-004 (`task_4_pr`) | LRU Cache + Serializer | hard | 3 | Missing move-to-end, missing eviction, None crash |
| INC-005 (`task_5_pr`) | Analytics Pipeline | very-hard | 3 | EMA alpha swap, population vs sample std, window boundary |
| INC-006 (`task_6_pr`) | Search Ranking Pipeline | expert | 3 | Coupled tokenizer/scorer/ranker — partial fix regresses tests |
| INC-007 (`task_7_pr`) | Config Service | expert | 3 | String→int types + `restart_service` gate |
| INC-008 (`task_8_pr`) | DB Migration | expert | 3 | Lexicographic sort + silent exceptions + `executescript` |
| INC-009 (`task_9_pr`) | Memory Leak / OOM | expert | 3 | Unbounded cache, missing eviction, module-level accumulator |
| INC-010 (`task_10_pr`) | Network Timeout Cascade | expert | 3 | Two-sided timeout trap, retry off-by-one, circuit breaker threshold |

### Task Descriptions

**INC-001 — User Registration** (`medium`)
Three security bugs: email validation accepts anything (missing `+` quantifier), password check uses `>` instead of `>=`, and passwords are hashed with MD5 instead of SHA-256. Seeded: minimum password length varies per episode.

**INC-002 — Order Pricing Engine** (`medium-hard`)
Three arithmetic bugs: `total_price` adds quantity instead of multiplying, discount applies in wrong direction (adds instead of subtracts), and tax multiplier is misapplied as an offset. Seeded: tax rate varies per episode.

**INC-003 — Async Payment Processor** (`hard`)
Three integration bugs: API key is hardcoded instead of read from environment, retry logic uses a synchronous sleep in an async function, and the retry loop runs `range(max_retries)` instead of `range(max_retries + 1)`. Seeded: max retries varies.

**INC-004 — LRU Cache + Serializer** (`hard`)
Three data-structure bugs: `get()` never calls `move_to_end()`, `put()` never evicts when capacity is exceeded, and the JSON serializer crashes on `None` values. Seeded: cache capacity varies.

**INC-005 — Analytics Pipeline** (`very-hard`)
Three algorithmic bugs: EMA `alpha` and `(1-alpha)` are swapped, standard deviation uses population (`ddof=0`) instead of sample (`ddof=1`), and the sliding window boundary is off by one. Seeded: EMA alpha and window size vary.

**INC-006 — Search Ranking Pipeline** (`expert`)
Three coupled bugs: tokenizer splits on whitespace instead of punctuation, scorer computes `len(query_terms)` instead of term intersection, and ranker sorts ascending instead of descending. **Trap**: fixing only the scorer without fixing the tokenizer causes a previously-passing test to regress, requiring agents to reason about coupling.

**INC-007 — Config Service** (`expert`)
Three type bugs: `TIMEOUT` and `MAX_WORKERS` are defined as string literals instead of integers, `RETRY_DELAY` is negative. After patching, agent must call `restart_service` to bring the service online — submitting without restart earns partial credit only. Seeded: timeout value varies.

**INC-008 — DB Migration** (`expert`)
Three migration bugs: version sort is lexicographic (`sorted()`) instead of numeric, exceptions in migration steps are silently swallowed, and multi-statement SQL blocks require `executescript()` not `execute()`. Agent should use `inspect_logs` to diagnose before patching. Seeded: number of migrations varies.

**INC-009 — Memory Leak / OOM** (`expert`)
Three memory bugs: the cache uses a plain `list` (unbounded), `add()` never evicts old entries when at capacity, and a module-level `_processed_log` list accumulates every processed item forever. Service starts in `oom_killed` state; agent must patch all three leaks, then call `restart_service`. Seeded: `CACHE_MAXSIZE` varies.

**INC-010 — Network Timeout Cascade** (`expert`)
Three network bugs with a two-sided trap: `CONNECT_TIMEOUT = 0.001` is too low (causes immediate failure), but setting it above 60.0 also fails the range test — agent must choose a value in `[0.1, 60.0]`. Additionally, `range(retries + 1)` is an off-by-one (should be `range(retries)`), and the circuit breaker opens after `>= 1` failure instead of `>= failure_threshold`. Seeded: `MAX_RETRIES` and `failure_threshold` vary.

---

## Reward Structure

### Step Reward (non-terminal)

Every `patch_file` call emits a dense delta reward:

```
reward = 0.01 + Δpass_rate × 0.88
```

Where `Δpass_rate = (tests_passing_after - tests_passing_before) / total_tests`.

- Regressing tests gives negative delta (minimum step reward: `-0.1`)
- Trap actions (e.g., premature `restart_service`) increment `trap_count` and affect terminal reward

### Terminal Reward (submit_fix / submit_review)

Six-component weighted sum at submission:

| Component | Weight | Formula |
|-----------|--------|---------|
| `test_quality` | 40% | `fraction of tests passing` |
| `diagnosis` | 15% | `keyword overlap(root_cause, ground_truth_keywords)` |
| `efficiency` | 15% | `min(optimal_steps / actual_steps, 1.0)` |
| `exploration` | 10% | `min(unique_action_types / 5, 1.0)` |
| `trap_avoidance` | 10% | `max(1.0 - 0.2 × trap_count, 0.0)` |
| `submit_credit` | 10% | `1.0` (fixed credit for reaching submit) |

**Gate rule**: all tests passing → floor terminal reward at `0.50`; any test failing → cap at `0.49`.

**Range**: terminal reward ∈ `[0.01, 0.99]`.

---

## Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `run_tests` | `action_type` | Run full pytest suite (or `path` for targeted run) |
| `read_file` | `action_type`, `path` | Read any workspace file cleanly |
| `search_codebase` | `action_type`, `pattern` | Grep regex across all `.py` files |
| `inspect_logs` | `action_type` | Read `service.log` / `migration.log` |
| `patch_file` | `action_type`, `target_file`, `new_content` | Overwrite source file (triggers delta reward) |
| `restart_service` | `action_type` | Run `app_init.py` to apply changes (tasks 7, 8, 9) |
| `submit_fix` | `action_type`, `root_cause` | Terminal action — triggers 6-component grading |
| `submit_review` | `action_type` | Backward-compatible alias for `submit_fix` |
| `execute_command` | `action_type`, `command` | Escape-hatch bash execution |

---

## Observation Space

Each step returns a `CodeObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task identifier |
| `context` | `str` | Incident/PR description shown to the agent |
| `available_files` | `List[str]` | Files present in the sandbox |
| `action_result` | `str` | Stdout/stderr of last action |
| `step_number` | `int` | Steps consumed so far |
| `done` | `bool` | True when episode has ended |
| `reward` | `float` | Reward earned this step |
| `logs` | `List[{ts, level, msg}]` | Structured log entries from last service operation |
| `service_status` | `Optional[Dict]` | Service health state (status, health details) |

---

## State Model

`GET /state` returns a `ReviewState`:

| Field | Description |
|-------|-------------|
| `episode_id` | UUID of current episode |
| `task_id` | Active task |
| `step_count` | Steps taken |
| `current_task_index` | Round-robin index (0→9→0) |
| `total_reward` | Highest reward achieved this episode |
| `done` | Episode ended |
| `system_health` | `crashed / degraded / running / healthy / unknown` |
| `progress_depth` | Number of causal state improvements made |
| `trap_count` | Actions that worsened or had no effect on system state |
| `optimal_steps` | Known optimal step count for active task |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `{"status": "healthy"}` |
| `/metadata` | GET | Environment name, description, version |
| `/schema` | GET | JSON schemas for action/observation/state |
| `/tasks` | GET | List of all task IDs |
| `/reset` | POST | Start new episode; optionally pass `{"task_id": "task_3_pr"}` |
| `/step` | POST | Submit one `AgentAction`, receive `StepResponse` |
| `/state` | GET | Full `ReviewState` for current episode |
| `/mcp` | POST | JSON-RPC 2.0 — `tools/list` and `tools/call` |

### Example: Reset and Step

```bash
# Start a new episode on task 7
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_7_pr"}' | python3 -m json.tool

# Run tests
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "run_tests"}' | python3 -m json.tool

# Patch a file
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "patch_file", "target_file": "config/settings.py", "new_content": "TIMEOUT = 45\nMAX_WORKERS = 4\nRETRY_DELAY = 1.0\n"}' | python3 -m json.tool

# Submit with diagnosis
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_fix", "root_cause": "TIMEOUT and MAX_WORKERS were string literals instead of int; RETRY_DELAY was negative"}' | python3 -m json.tool
```

### MCP tools/call Example

```bash
curl -s -X POST http://localhost:7860/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "run_tests",
      "arguments": {}
    }
  }' | python3 -m json.tool
```

---

## Run Locally

### With Docker

```bash
docker build -t codereview-env .
docker run -p 7860:7860 codereview-env
```

### Without Docker

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
export HF_TOKEN=hf_...
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
python inference.py
```

---

## Baselines

Measured on the full 10-task suite (10 episodes per task, temperature=0):

| Model | Avg Score | Tasks Solved (≥0.5) |
|-------|-----------|---------------------|
| Random actions | ~0.06 | 0/10 |
| Greedy patch-all | ~0.35 | 3/10 |
| Llama-3.3-70B-Instruct | ~0.52 | 6/10 |
| GPT-4o | ~0.61 | 7/10 |
| Claude Sonnet 4.5 | ~0.68 | 8/10 |

*Scores are approximate — exact numbers vary with seeding and API version. The benchmark is designed so no model saturates all 10 tasks.*

---

## Anti-Exploit Hardening

13 hardening tests (`tests/test_hardening.py`) cover:

1. Conftest injection — agent cannot write `conftest.py` to override fixtures
2. Test file modification — `tests/` directory is read-only
3. Path traversal — `../` escapes are blocked
4. `chmod` bypass — permission changes do not affect grading
5. Reward farming — repeated `patch_file` on already-passing tests gives zero delta
6. `PYTEST_CONTROL_NAMES` blocklist — internal pytest hooks cannot be overwritten
7. Symlink attacks — symlinks pointing outside the workspace are rejected
8. Empty patch — zero-byte `new_content` is rejected
9. Binary injection — non-UTF-8 content is rejected

---

## Spec Compliance

```
openenv validate --url http://localhost:7860
```

Validated endpoints: `/health`, `/metadata`, `/schema`, `/reset`, `/step`, `/state`, `/mcp`.

All 10 tasks pass `openenv validate` end-to-end.

---

## Project Structure

```
codereview-env/
├── server/
│   ├── app.py            # FastAPI routes, MCP endpoint
│   └── environment.py    # Episode logic, reward computation, trap detection
├── tasks.py              # 10 task factories + graders + diagnosis keywords
├── models.py             # Pydantic models (AgentAction, CodeObservation, ReviewState)
├── client.py             # Python client for CodeReviewEnv API
├── inference.py          # Reference agent (OpenAI-compatible LLM)
├── openenv.yaml          # OpenEnv spec manifest
├── tests/
│   └── test_hardening.py # 13 anti-exploit hardening tests
├── Dockerfile
└── requirements.txt
```

---

## License

MIT — see [LICENSE](LICENSE).
