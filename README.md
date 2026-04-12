---
title: CodeReview-Env V5
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

# CodeReview-Env V5

> **The execution-grounded software incident repair benchmark for RL agents.**
> An LLM agent is dropped into a real Linux sandbox with a failing CI/CD pipeline or a downed service. It runs real shell commands, writes real file patches, restarts real services, and is graded by real `pytest` — no simulation layer, no mocked test results.

**Quick stats**

| | |
|---|---|
| Tasks | 8 across 5 difficulty tiers |
| Action types | 8 (`run_tests`, `read_file`, `search_codebase`, `inspect_logs`, `patch_file`, `restart_service`, `submit_fix`, `execute_command`) |
| Instance diversity | Seeded parametric factories — unique problem per episode reset |
| Execution model | Real subprocess · real filesystem · real pytest oracle |
| Reward shape | Dense delta reward on every `patch_file` |
| Difficulty ceiling | tasks 5–8 not saturated by greedy patching |
| Stateful tasks | tasks 7–8 require `restart_service` to apply changes; test_service/migration state persists across steps |
| Anti-exploit hardening | 13 tests covering conftest injection, chmod bypass, path traversal, reward farming |
| Spec compliance | OpenEnv `openenv validate` — 8/8 PASS |

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.dev)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/Thowfiq23/CodeReview-Env)
[![openenv validate](https://img.shields.io/badge/openenv%20validate-8%2F8%20PASS-brightgreen)](#spec-compliance)
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
| `reward_range: [-0.1, 1.0]` | Security violations return −0.05; grader caps at 0.99 |
| `max_steps: 10` | Enforced server-side, episode auto-terminates |
| Dockerfile | [Dockerfile](Dockerfile) — `docker build && docker run` works |
| HF Space | Live, containerized, responds to all endpoints |

---

## Action Space

The agent has 8 typed tools. Each turn it outputs one JSON object — exactly one tool.

### Exploration Tools

#### `run_tests` — Targeted pytest (preferred over `execute_command` for tests)
```json
{"action_type": "run_tests"}
{"action_type": "run_tests", "path": "tests/test_service.py"}
```
Runs `pytest <path> -v --tb=short`. Returns `0.01` on all-pass, `0.005` on failure/timeout. 20-second timeout.

#### `read_file` — Clean file reader
```json
{"action_type": "read_file", "path": "config/settings.py"}
{"action_type": "read_file", "path": "db/migrator.py"}
```
Reads any workspace file cleanly. No shell required. Output truncated to 4,000 chars (first N retained). Returns `0.01`.

#### `search_codebase` — Regex grep across all `.py` files
```json
{"action_type": "search_codebase", "pattern": "TIMEOUT|MAX_WORKERS"}
{"action_type": "search_codebase", "pattern": "sorted\\("}
```
Runs `grep -rn --include=*.py` in the workspace. Output truncated to 3,000 chars. Returns `0.01`.

#### `inspect_logs` — Read service / migration logs
```json
{"action_type": "inspect_logs"}
```
Reads `service.log`, `migration.log`, and `app.log` if present (generated by `restart_service`). Returns last 2,000 chars of each log. Returns `0.01` if logs found, `0.005` if none.

### Modification Tools

#### `patch_file` — Write a fix to disk (triggers delta reward)
```json
{
  "action_type": "patch_file",
  "target_file": "config/settings.py",
  "new_content": "TIMEOUT = 45\nMAX_WORKERS = 6\nRETRY_DELAY = 0.5\n"
}
```
**Constraints:** Path traversal blocked. Writing to `tests/` blocked — oracle is read-only. Invalid Python syntax rejected before disk write (`py_compile` check).

#### `restart_service` — Apply changes (tasks 7 & 8 only)
```json
{"action_type": "restart_service"}
```
Runs `python app_init.py` in the workspace. For task_7 this validates config and writes `service_state.json`. For task_8 this runs SQL migrations and creates `app.db`. Returns `0.01` on exit-code 0, `0.005` on failure. Generates log files readable by `inspect_logs`.

### Terminal Actions

#### `submit_fix` — Final grader with root-cause analysis (preferred)
```json
{"action_type": "submit_fix", "root_cause": "TIMEOUT was str not int; sorted() was lexicographic not numeric."}
```

#### `submit_review` — Backward-compatible alias for `submit_fix`
```json
{"action_type": "submit_review", "summary": "Replaced MD5 with SHA-256."}
```

Both terminal actions run `pytest tests/ -v` and end the episode. `submit_fix` additionally records the `root_cause` in episode metadata.

**Schema:**

```python
class AgentAction(BaseModel):
    action_type: Literal[
        "execute_command", "read_file", "search_codebase",
        "patch_file", "run_tests", "inspect_logs",
        "restart_service", "submit_fix", "submit_review",
    ]
    command:     Optional[str]   # execute_command
    path:        Optional[str]   # read_file (file path) or run_tests (test path)
    pattern:     Optional[str]   # search_codebase
    target_file: Optional[str]   # patch_file
    new_content: Optional[str]   # patch_file
    root_cause:  Optional[str]   # submit_fix
    summary:     Optional[str]   # submit_review (compat)
```

#### `execute_command` — Escape hatch
```json
{"action_type": "execute_command", "command": "ls -la migrations/"}
```
Run any bash command. Use for edge cases not covered by the typed tools. 15-second timeout, 4,000-char truncation.

---

## Observation Space

Every `reset()` and `step()` returns a `CodeObservation`:

```python
class CodeObservation(BaseModel):
    task_id:         str        # "task_1_pr" … "task_8_pr"
    context:         str        # Full PR description — the agent's briefing
    available_files: List[str]  # Files present at episode start
    action_result:   str        # stdout/stderr of last command, or status message
    step_number:     int        # Steps consumed so far (1-indexed)
    done:            bool       # True when episode has ended
    reward:          float      # Reward earned this step [-0.05, 0.99]
```

The `context` field is the agent's only briefing — it describes the PR, the failing CI checks, and the files to investigate. It does **not** reveal the bugs directly; the agent must read the files to understand what is wrong.

---

## Reward Function

The reward is **dense and delta-based** — `patch_file` rewards are proportional to the *improvement* in pass rate over the best result so far, not the absolute pass rate. This prevents reward farming (re-patching the same fix) and gives RL algorithms gradient information throughout the trajectory.

| Action | Reward | Formula |
|--------|--------|---------|
| `run_tests` — all pass | `0.01` | Exploration signal |
| `run_tests` — some fail / timeout | `0.005` | Failed run, small signal |
| `read_file` | `0.01` | Clean read signal |
| `search_codebase` | `0.01` | Pattern found or not found |
| `inspect_logs` — logs present | `0.01` | Diagnostic signal |
| `inspect_logs` — no logs yet | `0.005` | Prompt to call restart_service first |
| `restart_service` — exit 0 | `0.01` | Service/migration applied |
| `restart_service` — failure | `0.005` | Config/code still broken |
| `execute_command` — exit 0 | `0.01` | Exploration signal |
| `execute_command` — non-zero / timeout | `0.005` | Failed command |
| `patch_file` — pass rate improves | `0.01 – 0.89` | `0.01 + Δpass_rate × 0.88`; `prev_pass_rate` advances |
| `patch_file` — no improvement or regression | `0.005` | No farming; regression not rewarded |
| `patch_file` — security violation | `−0.05` | Path traversal, test tampering, conftest injection |
| `submit_fix` / `submit_review` — all tests pass | `0.99` | Full credit, terminal |
| `submit_fix` / `submit_review` — partial pass | `0.01 – 0.98` | `0.01 + pass_rate × 0.97` |
| Max steps (10) exceeded | last action's reward | Episode terminates; no forced floor |

**The 0.89 cap on `patch_file`** creates an intentional submit incentive: even a perfect patch earns 0.89; the agent must call `submit_review` to reach 0.99. This penalises "patch and stall" strategies.

**Episode score** = `sum(rewards) / MAX_TOTAL_REWARD` — rewards accumulate across steps, so an efficient agent (fewer wasted steps) scores higher than a slow one that reaches the same final state.

**Security penalty (−0.05):** path traversal, writing to `tests/`, creating pytest control files (`conftest.py`, `pytest.ini`, etc.) all return an explicit negative reward. A 13-test hardening suite verifies every exploit vector is blocked.

---

## Tasks

### Task 1 — `task_1_pr` — User Registration — Medium

**Scenario:** *PR #42 "Add user registration"* — CI is red: email validation passes garbage addresses, passwords one character too short are accepted, and passwords are stored as MD5 hashes.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `auth/register.py` | Email regex uses `*` (zero-or-more) instead of `+` (one-or-more) for local part | Change `*` → `+` |
| `auth/register.py` | Password length check is `len(password) > 8` (rejects 8-char passwords) | Change `>` → `>=` |
| `auth/register.py` | `hashlib.md5` used for password storage | Replace with `hashlib.sha256` |

**Parametric:** Each episode randomises the email/password test cases from a pool, so the agent cannot memorise expected values — it must fix the algorithm.

---

### Task 2 — `task_2_pr` — Order Pricing — Medium-Hard

**Scenario:** *PR #88 "Order pricing engine"* — QA reports totals are wrong: quantity isn't multiplied, discounts inflate instead of reduce the price, and tax is added flat instead of compounded.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `orders/pricing.py` | Sums `item.price` without multiplying by `item.quantity` | `item.price * item.quantity` |
| `orders/pricing.py` | Discount applied as `total + discount_amount` (increases price) | `total - discount_amount` |
| `orders/pricing.py` | Tax applied additively: `total + tax_rate * subtotal` instead of multiplicatively | `total * (1 + tax_rate)` |

**Parametric:** Prices, quantities, discount %, and tax rate are randomised each episode. Expected totals are derived from the correct formula — the agent must fix the arithmetic, not hardcode values.

---

### Task 3 — `task_3_pr` — Async Payment Processor — Hard

**Scenario:** *PR #105 "Stripe integration"* — deployment blocked: secrets scanner flags a hardcoded live key, the async handler blocks the event loop, and the retry loop runs one extra attempt.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `payments/processor.py` | `STRIPE_KEY = 'sk_live_...'` hardcoded | `os.getenv('STRIPE_KEY')` |
| `payments/processor.py` | `def process_payment` (sync) calls `time.sleep` — blocks event loop | `async def` + `await asyncio.sleep` |
| `payments/processor.py` | `range(max_retries + 1)` produces one extra attempt | `range(max_retries)` |

**Grader:** The async fix is verified via AST analysis — `process_payment` must be an `AsyncFunctionDef` containing an `Await` node. String tricks won't pass. `max_retries` is randomised each episode.

---

### Task 4 — `task_4_pr` — LRU Cache — Hard

**Scenario:** *PR #201 "Add LRU cache layer"* — cache evicts the wrong entry and crashes on deserialising `None` values.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `cache/store.py` | `get()` reads the value but never calls `move_to_end()` — LRU order not maintained | Call `self._store.move_to_end(key)` after the read |
| `cache/store.py` | `put()` evicts an entry but the `popitem` is missing — cache grows without bound | `self._store.popitem(last=False)` when `len > capacity` |
| `cache/serializer.py` | `deserialize(None)` raises `TypeError` instead of returning `None` | Guard with `if data is None: return None` |

**Parametric:** Cache capacity and key names are randomised each episode.

---

### Task 5 — `task_5_pr` — Analytics Pipeline — Very Hard

**Scenario:** *PR #317 "Analytics pipeline"* — three independent algorithmic bugs: EMA weights swapped, anomaly detector uses biased population std causing false positives, and sliding window drops the last window.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `analytics/aggregator.py` | `ema = alpha * prev + (1-alpha) * new_value` — weights swapped | `alpha * new_value + (1-alpha) * prev` |
| `analytics/detector.py` | `variance / len(values)` — population std (ddof=0) gives false positives | Divide by `len(values) - 1` (sample std, ddof=1) |
| `analytics/window.py` | `range(n - size)` — off-by-one drops last window | `range(n - size + 1)` |

**Why this is very hard:** All three bugs are subtle algorithmic errors. The EMA swap looks plausible, the std denominator difference only manifests on small datasets, and the range boundary is a classic off-by-one. All test values are computed from the correct formula using a random seed — the agent must understand the mathematics, not pattern-match against known outputs.

---

### Task 6 — `task_6_pr` — Search Ranking (Coupled Trap) — Expert

**Scenario:** *PR #412 "Document search ranking"* — CI is red on three files in a search pipeline. One of the three bugs is a trap: fixing it in isolation causes a currently-passing test to regress.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `search/indexer.py` | `text.split()` — splits on whitespace only, punctuation sticks to tokens | `re.split(r'\W+', text.lower())` and filter empty strings |
| `search/scorer.py` | `matches / len(doc_terms)` — penalises long documents unfairly; full match never reaches 1.0 | Divide by `len(query_terms)` |
| `search/ranker.py` | `sorted(..., key=...)` — ascending order (lowest score first) | Add `reverse=True` |

**The coupled trap — why greedy patching fails:**

With both `scorer.py` and `ranker.py` bugs present, `test_rank_best_match_first` accidentally **passes**. The buggy scorer assigns a *lower* score to the verbose document (many terms, diluted fraction) than to the short imprecise document (fewer terms, higher fraction). Ascending sort then places the verbose doc first — the correct answer — by accident.

```
# With both bugs:
verbose_doc: buggy_score = q / (q + extras)  ← small  → sorted first by ascending
short_doc:   buggy_score = (q-1) / q          ← large  → sorted second

# Fix scorer alone (scores now correct, ascending sort still present):
verbose_doc: correct_score = 1.0  → sorted LAST by ascending  ← TEST FAILS
short_doc:   correct_score < 1.0  → sorted first

# Fix ranker alone (descending, buggy scores still present):
short_doc:   buggy_score = (q-1)/q ← larger  → sorted first  ← TEST FAILS

# Fix both together:
verbose_doc: correct_score = 1.0  → sorted first by descending  ← TEST PASSES
```

An agent that patches `scorer.py`, reruns pytest, and sees a new failure has discovered the coupling and must now fix both files in one step. Greedy per-file patching with pytest between each patch will loop: fix scorer → ranking breaks → fix ranker → scorer needed too → fix scorer again → passes.

**Parametric:** Word pool, query size (2–3 terms), and verbose document length (3–6 extra terms) are randomised each episode. The coupling invariant is verified by the factory — the trap always holds regardless of seed.

---

### Task 7 — `task_7_pr` — Service Config Type Errors (Restart Gate) — Expert

**Scenario:** *INC-031 "Request handler service is down"* — the service crashes on startup because `config/settings.py` contains three type/value bugs introduced in the last config edit.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `config/settings.py` | `TIMEOUT = "45"` — string literal causes `TypeError` in time arithmetic | `TIMEOUT = 45` (remove quotes) |
| `config/settings.py` | `MAX_WORKERS = "6"` — string literal causes `TypeError` in `range()` | `MAX_WORKERS = 6` (remove quotes) |
| `config/settings.py` | `RETRY_DELAY = -1` — negative value causes retry storms | `RETRY_DELAY = 0.5` (non-negative) |

**The restart gate:** After fixing the config, the agent must call `restart_service`. `app_init.py` validates the config types and values, then writes `service_state.json` with `status="running"`. `test_service_state_running` reads this file — it **cannot pass without the restart_service call** regardless of how well the config is fixed.

**Parametric:** `TIMEOUT` value, `MAX_WORKERS` value, and `RETRY_DELAY` negative value are randomised each episode. The tests check types and ranges, not specific values — the agent must understand the fix, not memorise it.

---

### Task 8 — `task_8_pr` — Database Migration Ordering (Inspect + Restart) — Expert

**Scenario:** *INC-047 "Database migrations failing silently after refactor"* — `db/migrator.py` has three bugs that prevent the schema from being applied correctly. `migration.log` is empty because exceptions are swallowed.

| File | Bug | Correct Fix |
|------|-----|-------------|
| `db/migrator.py` | `sorted(files)` sorts lexicographically: `"10_add_indexes.sql"` runs before `"2_create_posts.sql"` — posts table created before users → FK fails | `sorted(files, key=lambda f: int(f.split('_')[0]))` |
| `db/migrator.py` | `except Exception: pass` — migration failures are silently swallowed; `migration.log` shows nothing | Re-raise: `except Exception as e: logging.error(...); raise` |
| `db/migrator.py` | `conn.execute(sql)` — only executes one statement per call; `10_add_indexes.sql` has two `CREATE INDEX` statements, silently dropping the second | `conn.executescript(sql)` |

**Workflow the agent must follow:**
1. `inspect_logs` — `migration.log` shows what failed (empty if bug 2 is active)
2. Fix all three bugs in `db/migrator.py`
3. `restart_service` — runs `app_init.py` → runs migrations → creates `app.db`
4. Tests read `app.db` for tables and indexes

**Parametric:** User column name (`username`/`email`/`handle`/`nickname`) and content column name are randomised each episode. Index names depend on the seed.

---

## Difficulty Progression

| Task | Domain | Bugs | Required Actions | Difficulty |
|------|--------|------|-----------------|------------|
| `task_1_pr` | Security / Regex | 3 | patch + submit | Medium |
| `task_2_pr` | Arithmetic Logic | 3 | patch + submit | Medium-Hard |
| `task_3_pr` | Security + Async | 3 | patch + submit | Hard |
| `task_4_pr` | Data Structures | 3 | patch + submit | Hard |
| `task_5_pr` | Algorithms / Stats | 3 | patch + submit | Very Hard |
| `task_6_pr` | Search / Coupling | 3 | patch + submit | Expert |
| `task_7_pr` | Config / Service | 3 | patch + **restart_service** + submit | Expert |
| `task_8_pr` | DB Migrations | 3 | **inspect_logs** + patch + **restart_service** + submit | Expert |

All tasks use **parametric factories** — each episode reset generates a fresh problem instance from a random seed so the agent cannot memorise expected values across episodes. The dense delta reward gives an RL agent a learning signal at every step, not just on success.

**Tasks 7 and 8 introduce a new difficulty axis: stateful system operations.** Fixing the code is necessary but not sufficient — the agent must also trigger side effects (service restart, migration run) that persist state across subsequent steps. This tests whether an agent can reason about multi-step causal chains beyond just code edits.

---

## Baseline Results

**Model:** `meta-llama/Llama-3.3-70B-Instruct` via HF Router (inference.py default)
**Temperature:** `0.0` (fully deterministic — results are exactly reproducible)
**Infrastructure:** 2 vCPU / 4 GB RAM (well within the 8 GB limit)
**Runtime:** ~15 minutes for all 8 tasks · Model: `meta-llama/Llama-3.3-70B-Instruct` · Temp: `0.0`

### Per-task score table

| Task | Difficulty | Llama-3.3-70B score | Notes |
|------|-----------|-------------------|-------|
| `task_1_pr` | Medium | **0.99** | Solved reliably; regex + hash bugs well within model knowledge |
| `task_2_pr` | Medium-Hard | **0.99** | Parametric prices prevent memorisation; formula bugs are findable |
| `task_3_pr` | Hard | **0.99** | AST-verified async fix; step 5 shows semantic rejection (`def` without `async`) |
| `task_4_pr` | Hard | **0.99** | LRU order semantics require understanding `OrderedDict.move_to_end` |
| `task_5_pr` | Very Hard | **~0.85*** | EMA + std + window bugs are algorithmic; not reliably solved in 10 steps |
| `task_6_pr` | Expert | **~0.70*** | Coupled trap: greedy patching stalls; requires multi-file reasoning |
| `task_7_pr` | Expert | **~0.60*** | Config type bugs are findable, but restart_service gate requires multi-step planning |
| `task_8_pr` | Expert | **~0.55*** | Migration order + silent exceptions + executescript; inspect_logs + restart_service chain required |

*Estimated from grader probes — full multi-seed baseline run pending. Tasks 5–8 are not saturated.*

### task_6_pr — the coupling trap in detail

The reward trajectory for the most natural greedy path through task_6 (measured):

| Step | Action | Pass rate | Step reward | What happened |
|------|--------|-----------|-------------|---------------|
| 0 | — (episode start) | 5/7 = 0.71 | — | `test_rank_best_match_first` passes by accident |
| 1 | `patch_file scorer.py` | 5/7 = 0.71 | **0.6386** | Delta 0→0.71 earns positive reward, but ranking test **regresses** |
| 2 | `patch_file ranker.py` | 6/7 = 0.86 | **0.1358** | Delta 0.71→0.86; coupling resolved, tokenizer still wrong |
| 3 | `patch_file indexer.py` | 7/7 = 1.0 | **0.1358** | Delta 0.86→1.0; all bugs fixed |
| 4 | `submit_review` | — | **0.99** | Full suite green; episode ends |

The trap is not a missing reward signal — step 1 earns a healthy 0.64. The problem is that after step 1, a pytest run reveals `test_rank_best_match_first` has regressed. An agent that submits at this point scores ≈ 0.70 (partial credit). One that reads the regression, understands it requires fixing `ranker.py` too, and continues earns full credit — but it must resist the urge to submit after the first positive reward.

> **RL signal note:** Step 5 of task_3 demonstrates the grader enforcing semantic correctness: a syntactically valid `def process_payment` (without `async`) earns floor reward — the AST check rejects it. The agent self-corrects on step 6 to `async def` and earns `0.89`. The same grader strictness applies across all tasks.

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
export HF_TOKEN=your_hf_api_key
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct  # default
export API_BASE_URL=https://router.huggingface.co/v1  # default
export ENV_URL=http://localhost:7860                  # or the HF Space URL

python3 inference.py
```

### Run Against the Live HF Space

```bash
export ENV_URL=https://thowfiq23-codereview-env.hf.space
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

# Start a new episode (cycles through tasks 1 → 2 → … → 8 → 1 → …)
curl -X POST http://localhost:7860/reset

# Run the full test suite
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "run_tests"}'

# Read a source file
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "read_file", "path": "config/settings.py"}'

# Write a file fix (triggers partial reward)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "patch_file", "target_file": "config/settings.py", "new_content": "TIMEOUT = 45\nMAX_WORKERS = 6\nRETRY_DELAY = 0.5\n"}'

# Apply changes (tasks 7-8: run app_init.py)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "restart_service"}'

# Submit and grade (terminal — episode ends)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_fix", "root_cause": "Config values were strings not ints."}'

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
    current_task_index:  int    # Position in task cycle (0–7)
    total_reward:        float  # Highest reward achieved this episode (any action type)
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
version: 2.0.0
description: >
  Execution-grounded software incident repair benchmark. Agents fix real Python
  bugs and system failures in an isolated sandbox; real pytest grades every patch.
tasks:
  - task_1_pr  # medium: user registration
  - task_2_pr  # medium-hard: order pricing
  - task_3_pr  # hard: async payment
  - task_4_pr  # hard: LRU cache
  - task_5_pr  # very-hard: analytics pipeline
  - task_6_pr  # expert: coupled search ranking
  - task_7_pr  # expert: service config + restart gate
  - task_8_pr  # expert: DB migration + inspect_logs + restart gate
action_model: AgentAction
observation_model: CodeObservation
state_model: ReviewState
max_steps: 10
reward_range: [-0.1, 1.0]
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

# 3. Verify all 8 tasks produce scores in [0.0, 1.0]
python3 inference.py

# 4. Docker build
docker build -t codereview-env .
docker run -p 7860:7860 codereview-env
```

Expected output for `openenv validate --url`:

```json
{"passed": true, "summary": {"passed_count": 8, "total_count": 8, "failed_criteria": []}}
```
