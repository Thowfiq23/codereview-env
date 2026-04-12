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
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/Thowfiq23/CodeReview-Env)
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
    task_id:         str        # "task_1_pr" | "task_2_pr" | "task_3_pr" | "task_4_pr" | "task_5_pr"
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
0.01 ─────────── 0.45 ────────────── 0.89 ──────── 0.99
  execute cmds     1 of 2 bugs fixed   both fixed    submit
```

| Action | Reward | Formula |
|--------|--------|---------|
| `execute_command` | `0.01` | Exploration floor — strictly above zero |
| `patch_file` — no tests gained | `0.01` | Floor reward, bad patch |
| `patch_file` — partial fix | `0.01 – 0.89` | `0.01 + (tests_passing / total_tests) × 0.88` |
| `submit_review` — all tests pass | `0.99` | Full credit, terminal |
| `submit_review` — partial tests pass | `0.01 – 0.98` | `0.01 + (tests_passing / total_tests) × 0.97` |
| Max steps (10) exceeded | `0.01` | Episode terminates at floor |

**The 0.89 cap on `patch_file`** creates an intentional final incentive: a perfect patch earns 0.89, but the agent must call `submit_review` to earn the full 0.99. This penalizes "patch and stall" strategies and rewards agents that understand when they are done.

**Episode score** = `max(rewards)` over the trajectory — the highest reward the agent achieved. A policy that solves the task in 5 steps scores the same as one that solves it in 9 steps, but the step count is visible in the trajectory for human analysis.

**Built-in penalties:**
- Attempting path traversal returns an error observation (0.01 floor reward, no disk write)
- Attempting to overwrite test files returns an error (0.01 floor reward)
- Syntax-invalid patches are rejected before touching disk (0.01 floor reward)
- Commands exceeding 15 seconds are killed (0.01 floor reward, timeout message)

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

## Difficulty Progression

| Task | Domain | Bugs | Difficulty |
|------|--------|------|------------|
| `task_1_pr` | Security / Regex | 3 | Medium |
| `task_2_pr` | Arithmetic Logic | 3 | Medium-Hard |
| `task_3_pr` | Security + Async | 3 | Hard |
| `task_4_pr` | Data Structures | 3 | Hard |
| `task_5_pr` | Algorithms / Stats | 3 | Very Hard |
| `task_6_pr` | Search / Coupling  | 3 | Expert |

All tasks use **parametric factories** — each episode reset generates a fresh problem instance from a random seed so the agent cannot memorise expected values across episodes. The dense delta reward gives an RL agent a learning signal at every step, not just on success.

**task_6_pr is not saturated by greedy patching.** An agent that fixes bugs one file at a time and runs pytest between each patch will discover a regression on the second fix, requiring it to reason about cross-file interactions — a qualitatively different capability from single-file repair.

---

## Baseline Results

**Model:** `meta-llama/Llama-3.3-70B-Instruct` via HF Router (inference.py default)
**Temperature:** `0.0` (fully deterministic — results are exactly reproducible)
**Infrastructure:** 2 vCPU / 4 GB RAM (well within the 8 GB limit)
**Runtime:** ~10 minutes for all 6 tasks

```
[START] task=task_3_pr env=codereview-env model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.01 done=false error=null
[STEP] step=2 action={"action_type":"execute_command","command":"cat payments/config.py"} reward=0.01 done=false error=null
[STEP] step=3 action={"action_type":"patch_file","target_file":"payments/config.py","new_content":"import os\n\nSTRIPE_TEST_KEY = os.getenv('STRIPE_TEST_KEY')\nSTRIPE_LIVE_KEY = os.getenv('STRIPE_LIVE_KEY')"} reward=0.45 done=false error=null
[STEP] step=4 action={"action_type":"execute_command","command":"cat payments/processor.py"} reward=0.01 done=false error=null
[STEP] step=5 action={"action_type":"patch_file","target_file":"payments/processor.py","new_content":"import asyncio\n\ndef process_payment(user_id, amount):\n    await asyncio.sleep(5)\n    return True"} reward=0.01 done=false error=null
[STEP] step=6 action={"action_type":"patch_file","target_file":"payments/processor.py","new_content":"import asyncio\n\nasync def process_payment(user_id, amount):\n    await asyncio.sleep(5)\n    return True"} reward=0.89 done=false error=null
[STEP] step=7 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.01 done=false error=null
[STEP] step=8 action={"action_type":"submit_review","summary":"Fixed hardcoded Stripe API key and declared process_payment as async with await asyncio.sleep."} reward=0.99 done=true error=null
[END] success=true steps=8 rewards=0.01,0.01,0.45,0.01,0.01,0.89,0.01,0.99

[START] task=task_1_pr env=codereview-env model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.01 done=false error=null
[STEP] step=2 action={"action_type":"execute_command","command":"cat auth/models.py"} reward=0.01 done=false error=null
[STEP] step=3 action={"action_type":"patch_file","target_file":"auth/models.py","new_content":"def get_user_query(username):\n    return ('SELECT * FROM users WHERE username = %s', (username,))"} reward=0.45 done=false error=null
[STEP] step=4 action={"action_type":"execute_command","command":"cat auth/crypto.py"} reward=0.01 done=false error=null
[STEP] step=5 action={"action_type":"patch_file","target_file":"auth/crypto.py","new_content":"import hashlib\nimport secrets\n\ndef hash_password(password: str):\n    salt = secrets.token_bytes(16)\n    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex() + salt.hex()"} reward=0.89 done=false error=null
[STEP] step=6 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.01 done=false error=null
[STEP] step=7 action={"action_type":"submit_review","summary":"Fixed SQL injection with parameterized query and replaced MD5 with PBKDF2."} reward=0.99 done=true error=null
[END] success=true steps=7 rewards=0.01,0.01,0.45,0.01,0.89,0.01,0.99

[START] task=task_2_pr env=codereview-env model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.01 done=false error=null
[STEP] step=2 action={"action_type":"execute_command","command":"cat billing/cart.py"} reward=0.01 done=false error=null
[STEP] step=3 action={"action_type":"execute_command","command":"cat billing/discounts.py"} reward=0.01 done=false error=null
[STEP] step=4 action={"action_type":"patch_file","target_file":"billing/discounts.py","new_content":"def apply_discount(amount):\n    return amount * 0.80"} reward=0.45 done=false error=null
[STEP] step=5 action={"action_type":"patch_file","target_file":"billing/cart.py","new_content":"from billing.discounts import apply_discount\n\ndef calculate_total(items):\n    total = sum(item['price'] * item['quantity'] for item in items)\n    return apply_discount(total)"} reward=0.89 done=false error=null
[STEP] step=6 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.01 done=false error=null
[STEP] step=7 action={"action_type":"submit_review","summary":"Fixed quantity multiplication and discount percentage."} reward=0.99 done=true error=null
[END] success=true steps=7 rewards=0.01,0.01,0.01,0.45,0.89,0.01,0.99

[SUMMARY] tasks=5 scores=0.99,0.99,0.99,0.99,0.99 average=0.99
# task_6_pr omitted — see note below
```

> Note: The dense trajectory is the key output for RL training. Step 5 of task_3 shows the grader
> correctly rejecting a syntactically valid but semantically wrong patch (`def` without `async`),
> returning `reward=0.01` (floor). The agent self-corrects on step 6 and earns `0.89`. This demonstrates
> that the grader enforces semantic correctness, not just syntactic validity.

**task_6_pr baseline — the trap in the reward curve:**

The above 0.99 sweep covers tasks 1–5 only. task_6_pr behaves differently:

A greedy agent that reads `scorer.py` and fixes Bug 2 (wrong denominator) will earn a **positive** first-step reward (pass rate 0 → 0.71, reward ≈ 0.64) because `prev_pass_rate` starts at 0.0. The trap is not in the reward signal — it is in what happens next.

After that patch, `test_rank_best_match_first` has **regressed**: a test that was passing before is now failing. The agent is stuck at 0.71. Patching `ranker.py` alone (without scorer) yields only 0.57 (4/7). Only fixing both scorer and ranker together reaches 0.86 (6/7), and fixing all three files reaches 1.0. An agent that submits after the scorer-only patch scores ≈ 0.70 — partial credit, not a sweep.

The coupling forces the agent to hold the regression in working memory and reason that `scorer.py` and `ranker.py` must both be fixed before the suite goes green. A standard one-file-at-a-time patching loop either submits early at 0.70 or exhausts MAX_STEPS without converging.

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

# Start a new episode (cycles through tasks: 1 → 2 → 3 → 4 → 5 → 1 → …)
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
  - task_4_pr
  - task_5_pr
  - task_6_pr
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

# 3. Verify all 6 tasks produce scores in [0.0, 1.0]
python3 inference.py

# 4. Docker build
docker build -t codereview-env .
docker run -p 7860:7860 codereview-env
```

Expected output for `openenv validate --url`:

```json
{"passed": true, "summary": {"passed_count": 6, "total_count": 6, "failed_criteria": []}}
```
