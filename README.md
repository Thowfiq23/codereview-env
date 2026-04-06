---
title: CodeReview-Env V4
emoji: 🤖
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

**An OpenEnv-compliant autonomous software engineering benchmark.**

An LLM agent is placed inside a real Linux sandbox and given a failing CI/CD pipeline. It must investigate the repository, identify the bugs, patch the source files on disk, and verify its own fixes by re-running the test suite. The episode ends when the agent calls `submit_review` — at which point the environment runs `pytest` one final time to produce the score.

This is the same evaluation loop used by SWE-bench: the agent writes real code, and a real test runner decides if it was correct.

---

## Why This Domain

Code review and bug triage happen thousands of times per day across every software team. An AI agent capable of reliably diagnosing a failing test, tracing the bug to its source file, and writing a correct fix has immediate, measurable value: faster CI cycles, fewer regression escapes, lower on-call burden.

Existing benchmarks for this domain (SWE-bench, HumanEval) are either too expensive to run interactively or produce only a pass/fail signal with no intermediate gradient. This environment fills that gap: it is fully interactive, containerized, and produces a dense reward trajectory that RL algorithms can learn from.

---

## Architecture

```
Agent (inference.py)
  |
  | HTTP/JSON  (OpenEnv API)
  v
FastAPI Server  :7860
  |
  | subprocess.run() / open()
  v
Physical Sandbox  /tmp/codereview_workspaces/{episode_uuid}/
  ├── auth/models.py          <-- buggy source code, written to disk on reset()
  ├── auth/crypto.py
  └── tests/test_auth.py      <-- deterministic test oracle
```

Each episode gets a fresh, UUID-named workspace on disk. The server writes the task's repository files at `reset()`, the agent reads and mutates them via tools, and the grader calls `pytest` directly against the agent's modified files.

---

## OpenEnv Specification Compliance

| Requirement | Implementation |
|---|---|
| Typed `Action` Pydantic model | `AgentAction` in [models.py](models.py) |
| Typed `Observation` Pydantic model | `CodeObservation` in [models.py](models.py) |
| Typed `Reward` Pydantic model | `EpisodeReward` in [models.py](models.py) |
| Typed `State` Pydantic model | `ReviewState` in [models.py](models.py) |
| `POST /reset` | Returns clean `CodeObservation`, deploys sandbox |
| `POST /step` | Returns observation, reward, done, info |
| `GET /state` | Returns current `ReviewState` |
| `GET /tasks` | Enumerates all 3 task IDs |
| `GET /health` | Liveness probe |
| `openenv.yaml` manifest | [openenv.yaml](openenv.yaml) |
| `reward_range: [0.0, 1.0]` | Enforced by grader |
| `max_steps: 10` | Enforced server-side — episode auto-terminates |
| Dockerfile | [Dockerfile](Dockerfile) — builds and runs on port 7860 |
| HF Space deployment | Live and responding to `/reset` |

---

## Action Space

The agent has three tools. Each turn it outputs exactly one JSON object.

**Tool 1 — `execute_command`**

Runs a bash command in the sandbox. The agent uses this to run `pytest`, read files, list directories, or inspect error output.

```json
{"action_type": "execute_command", "command": "pytest tests/ -v"}
{"action_type": "execute_command", "command": "cat auth/models.py"}
```

**Tool 2 — `patch_file`**

Overwrites a file on disk with the agent's proposed fix. Triggers an immediate partial reward based on how many tests now pass.

```json
{
  "action_type": "patch_file",
  "target_file": "auth/crypto.py",
  "new_content": "import hashlib\n\ndef hash_password(password: str):\n    return hashlib.sha256(password.encode()).hexdigest()"
}
```

**Tool 3 — `submit_review`**

Declares the episode complete. The environment runs `pytest` one final time against all patched files and assigns the terminal reward. Agents should only call this after verifying tests pass.

```json
{"action_type": "submit_review", "summary": "Replaced MD5 with SHA-256 and parameterized the SQL query."}
```

---

## Observation Space

Every `step()` and `reset()` call returns a `CodeObservation`:

```python
class CodeObservation(BaseModel):
    task_id:         str        # Identifies the current task, e.g. "task_1_pr"
    context:         str        # PR description shown to the agent
    available_files: List[str]  # Files present in the sandbox at episode start
    action_result:   str        # Full stdout/stderr of the last command, or status message
    step_number:     int        # Steps taken so far in this episode
    done:            bool       # True when the episode has ended
    reward:          float      # Reward earned this step, in [0.0, 1.0]
```

---

## Reward Function

The reward is **dense**: the agent receives signal at every `patch_file` call, not only at episode end. This ensures an RL training loop has gradient information throughout the trajectory.

| Action | Reward | Formula |
|---|---|---|
| `execute_command` | `0.0` | Exploration is free and unlimited |
| `patch_file` (no tests fixed) | `0.0` | |
| `patch_file` (partial fix) | `0.0 – 0.9` | `(tests_passing / total_tests) x 0.9` |
| `submit_review` (all pass) | `1.0` | Full credit, episode ends |
| `submit_review` (partial) | `0.0 – 1.0` | `tests_passing / total_tests` |
| Max steps exceeded | `0.0` | Episode terminates with no additional reward |

The 0.9 cap on `patch_file` creates an intentional incentive gradient: a perfect patch earns 0.9 but the agent must still call `submit_review` to confirm the fix and earn the full 1.0. Agents that patch correctly and submit immediately score better than agents that patch correctly but stall.

**Penalties built into the environment design:**

- Path traversal attempts (`../../../etc/passwd`) are blocked and return a security error
- Commands that run longer than 15 seconds are killed and return a timeout error
- Episodes that exceed 10 steps are terminated server-side with no reward
- Calling `submit_review` before tests pass results in partial credit only

---

## Tasks

### Task 1 — `task_1_pr` — Security — Easy to Medium

**Scenario:** PR #42 "Add user authentication" — the security scanner is blocking the merge.

Two vulnerabilities in the new auth module:

| File | Bug | Required Fix |
|---|---|---|
| `auth/models.py` | `get_user_query` builds SQL via f-string interpolation — SQL injection | Return a parameterized query with `%s` placeholder |
| `auth/crypto.py` | `hash_password` uses `hashlib.md5` — broken since 2004 | Replace with SHA-256, bcrypt, or PBKDF2 |

The test suite checks structural properties of the code (not runtime values), so the agent must understand what constitutes a secure pattern, not just copy a template.

**Expected difficulty for a capable frontier model:** Solvable in 5–8 steps.

---

### Task 2 — `task_2_pr` — Logic — Medium

**Scenario:** PR #88 "Implement shopping cart billing" — checkout totals are wrong.

Two arithmetic bugs in the billing module:

| File | Bug | Required Fix |
|---|---|---|
| `billing/cart.py` | `calculate_total` sums `item['price']` without multiplying by `item['quantity']` | Multiply price by quantity |
| `billing/discounts.py` | `apply_discount` returns `amount * 0.20` — charges 20% instead of retaining 80% | Return `amount * 0.80` |

A cart with two $10 items and one $20 item should total $32 after a 20% discount. The buggy code returns $10. The agent must trace through the math, not just find keywords.

**Expected difficulty for a capable frontier model:** Solvable in 5–7 steps. Reliably solved in testing.

---

### Task 3 — `task_3_pr` — Security + Async Performance — Hard

**Scenario:** PR #105 "Integrate Stripe payment processor" — deployment is blocked by a secrets scan and a performance regression.

Two issues requiring different domain knowledge:

| File | Bug | Required Fix |
|---|---|---|
| `payments/config.py` | `STRIPE_LIVE_KEY = 'sk_live_9876543210qwerty'` hardcoded | Move to `os.getenv('STRIPE_LIVE_KEY')` |
| `payments/processor.py` | `time.sleep(5)` called synchronously in a payment handler — blocks the event loop | Replace with `async def` + `await asyncio.sleep(5)` |

This task requires knowledge of both secrets management and Python's async/await model simultaneously. Frontier models often fix one bug but miss the semantic difference between `def` and `async def`.

**Expected difficulty for a capable frontier model:** Partially solvable in 10 steps; full solution requires careful handling of both issues.

---

## Difficulty Progression Summary

| Task | Domain | Bugs | Baseline Score (Llama-3.3-70B) |
|---|---|---|---|
| `task_1_pr` | Security | 2 | 1.00 |
| `task_2_pr` | Arithmetic Logic | 2 | 1.00 |
| `task_3_pr` | Security + Async | 2 | 0.45 (partial) |

Task 3 genuinely challenges frontier models, satisfying the rubric requirement that the hard task not be trivially solvable.

---

## Baseline Results

Model: `meta-llama/Llama-3.3-70B-Instruct` via Groq API (temperature 0.1)

```
[START] task=task_1_pr env=codereview-env model=llama-3.3-70b-versatile
[STEP] step=1 action={"action_type":"execute_command","command":"pytest tests/"} reward=0.00 done=false error=null
[STEP] step=3 action={"action_type":"patch_file","target_file":"auth/models.py",...} reward=0.45 done=false error=null
[STEP] step=5 action={"action_type":"patch_file","target_file":"auth/crypto.py",...} reward=0.90 done=false error=null
[STEP] step=7 action={"action_type":"submit_review",...} reward=1.00 done=true error=null
[END] success=true steps=7 score=1.00 rewards=0.00,0.00,0.45,0.00,0.90,0.00,1.00

[START] task=task_2_pr env=codereview-env model=llama-3.3-70b-versatile
[STEP] step=4 action={"action_type":"patch_file","target_file":"billing/discounts.py",...} reward=0.45 done=false error=null
[STEP] step=5 action={"action_type":"patch_file","target_file":"billing/cart.py",...} reward=0.90 done=false error=null
[STEP] step=7 action={"action_type":"submit_review",...} reward=1.00 done=true error=null
[END] success=true steps=7 score=1.00 rewards=0.00,0.00,0.00,0.45,0.90,0.00,1.00

[START] task=task_3_pr env=codereview-env model=llama-3.3-70b-versatile
[STEP] step=7 action={"action_type":"patch_file","target_file":"payments/processor.py",...} reward=0.45 done=false error=null
[STEP] step=10 action={"action_type":"patch_file","target_file":"payments/processor.py",...} reward=0.45 done=false error=null
[END] success=false steps=10 score=0.45 rewards=0.00,0.00,0.00,0.00,0.00,0.00,0.45,0.00,0.00,0.45

[SUMMARY] tasks=3 scores=1.00,1.00,0.45 average=0.82
```

The 0.45 on task_3 reflects the dense reward working correctly: the agent fixed the async bug (earning partial credit), but ran out of steps before removing the hardcoded API key. An RL agent training on this trajectory has a clear signal to optimize: fix both bugs before step 10.

---

## Setup and Usage

### Run with Docker

```bash
git clone https://github.com/Thowfiq23/codereview-env
cd codereview-env

docker build -t codereview-env .
docker run -p 7860:7860 codereview-env
```

### Run Locally

```bash
pip install -r requirements.txt
PYTHONPATH=$PWD uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run the Baseline Agent

```bash
export HF_TOKEN=your_api_key
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_URL=http://localhost:7860

python inference.py
```

The `ENV_URL` variable accepts any running instance of the server, including the deployed HF Space URL.

### API Quick Reference

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Start an episode
curl -X POST http://localhost:7860/reset

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "execute_command", "command": "pytest tests/"}'

# Check episode state
curl http://localhost:7860/state
```

---

## Security Properties

| Threat | Mitigation |
|---|---|
| Path traversal via `patch_file` | Any `..` in `target_file` is rejected before touching disk |
| Hanging commands | 15-second subprocess timeout, then `TimeoutExpired` is caught cleanly |
| Unbounded episodes | Server enforces `max_steps=10` and terminates the episode |
| Cross-episode contamination | Fresh UUID workspace per `reset()`, previous workspace deleted |
| Server credential leakage via `env` command | Subprocess receives a minimal clean environment — no API keys, no HF tokens |
| Disk exhaustion from crashed episodes | Workspace deleted after `submit_review`; cleaned up on next `reset()` |

---

## Project Structure

```
codereview-env/
├── server/
│   ├── app.py            FastAPI application, all HTTP endpoints
│   └── environment.py    Sandbox engine: workspace management, tool execution, grading
├── models.py             Pydantic schemas: AgentAction, CodeObservation, EpisodeReward, ReviewState
├── tasks.py              Task definitions with embedded source files and test files
├── client.py             Thin HTTP client used by inference.py
├── inference.py          ReAct agent loop with mandatory [START]/[STEP]/[END] logging
├── openenv.yaml          OpenEnv spec manifest
├── requirements.txt      Python dependencies
└── Dockerfile            Container definition, installs all dependencies from requirements.txt
```
