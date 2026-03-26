# CodeReview-Env 🔍

> An OpenEnv environment where an AI agent reviews Python pull-request diffs
> and must identify bugs, security vulnerabilities, and logic errors — just
> like a senior engineer doing a real code review.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why This Environment?

Every software team does code review. Bad reviews miss real bugs, security
flaws, and logic errors that cost hours of debugging or, worse, create
production incidents. Training an RL agent to review code well has direct,
measurable value — and the grader is fully deterministic, making it ideal
for reproducible RL training.

---

## Environment Overview

| Property | Value |
|---|---|
| Domain | Software engineering / security |
| Language | Python |
| Tasks | 3 (easy → medium → hard) |
| Max steps per episode | 3 (one per task) |
| Reward range | 0.0 – 1.0 per step |
| Grader type | Fully deterministic (no LLM judge) |
| API | OpenEnv spec: `reset()` / `step()` / `state` |

---

## Observation Space

Each step the agent receives a `CodeObservation`:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Unique task identifier |
| `task_description` | `str` | Natural-language instruction |
| `code_snippet` | `str` | Python code to review |
| `filename` | `str` | Simulated file name |
| `language` | `str` | Always `"python"` |
| `context` | `str` | Background on what the code does |
| `step_number` | `int` | Current step in episode |
| `feedback` | `str \| null` | Grader feedback from previous step |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward from the previous step |

---

## Action Space

The agent submits a `ReviewAction`:

```json
{
  "comments": [
    {
      "line_number": 7,
      "issue_type": "bug",
      "severity": "high",
      "description": "Loop starts at index 1 instead of 0, skipping the first element.",
      "suggested_fix": "Change range(1, len(numbers)) to range(len(numbers))"
    }
  ],
  "summary": "Found one off-by-one bug in the loop range."
}
```

### Field constraints

| Field | Type | Allowed values |
|---|---|---|
| `line_number` | `int` | ≥ 1 |
| `issue_type` | `str` | `bug` · `security` · `style` · `performance` · `logic` |
| `severity` | `str` | `low` · `medium` · `high` · `critical` |
| `description` | `str` | min 10 characters |
| `suggested_fix` | `str \| null` | optional |

---

## Tasks

### Task 1 — Bug Detection (Easy)

**File:** `data_processor.py`

A sensor-data aggregation pipeline has a single off-by-one bug in a loop.
The agent must find it, classify it as a `bug`, and label it `high` severity.

- Ground truth issues: **1**
- Expected score for a perfect review: **1.00**
- Expected score for a blank review: **0.00**

---

### Task 2 — Security Audit (Medium)

**File:** `user_auth.py`

A user-authentication module for a 50 000-account web app contains three
security issues: a hardcoded secret key, a SQL injection vulnerability, and
use of the broken MD5 algorithm for password hashing.

- Ground truth issues: **3**
- Expected score for a perfect review: **1.00**
- Expected score for finding 2/3 with correct severity: **~0.58**

---

### Task 3 — Full PR Review (Hard)

**File:** `payment_processor.py`

A payment-processing module ready for production deployment contains five
mixed issues: a hardcoded live API key, missing input validation, a missing
HTTP timeout, sensitive data in plaintext logs, and floating-point arithmetic
for monetary calculations.

- Ground truth issues: **5**
- Expected score for a perfect review: **1.00**
- Expected score for finding 3/5 with correct severity: **~0.45**

---

## Reward Function

The grader is **fully deterministic** — no LLM judge.

An agent comment **matches** a ground-truth issue when all three criteria are met:

1. `line_number` is within ±3 of the ground-truth line
2. `issue_type` matches the ground-truth type
3. At least one keyword from the ground-truth keyword list appears in the
   agent's `description` (case-insensitive)

### Scoring formula

```
recall_score      = issues_matched / total_real_issues        (weight 0.50)
severity_accuracy = correct_severities / issues_matched       (weight 0.25)
fp_penalty        = min(false_positives / total_issues, 1.0)  (weight 0.25, subtracted)

total = max(0.0, min(1.0,
    0.50 * recall_score
  + 0.25 * severity_accuracy
  - 0.25 * fp_penalty
))
```

### Why this reward is useful for RL

- **Continuous signal across the trajectory** — partial credit for every
  issue found, not just binary pass/fail.
- **Penalises false positives** — the agent learns to be precise, not just
  verbose.
- **Rewards severity accuracy** — the agent learns that not all bugs are equal.
- **Reproducible** — same action always produces the same score.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode, returns Task 1 observation |
| `POST` | `/step` | Submit a `ReviewAction`, returns next observation + reward |
| `GET` | `/state` | Current episode state |
| `GET` | `/tasks` | List all tasks + full action schema |
| `POST` | `/grader` | Score a `ReviewAction` against any task |
| `POST` | `/baseline` | Run baseline inference script, returns per-task scores |
| `GET` | `/health` | Health check |

---

## Setup & Usage

### Option A — Run with Docker (recommended)

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/your-hf-username/codereview-env
cd codereview-env

# 2. Build the image
docker build -f server/Dockerfile -t codereview-env .

# 3. Run (set your API key for baseline)
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key_here \
  codereview-env
```

The server will be live at `http://localhost:7860`.

---

### Option B — Run locally without Docker

```bash
# 1. Install dependencies
pip install openenv-core fastapi uvicorn pydantic openai requests

# 2. Install this package in editable mode
pip install -e .

# 3. Start the server
uvicorn codereview_env.server.app:app --host 0.0.0.0 --port 7860 --reload
```

---

### Quick interaction example

```python
import asyncio
from codereview_env import ReviewAction, ReviewComment
# If using the remote HF Space:
from openenv.core.env_client import EnvClient

async def main():
    async with EnvClient(base_url="https://your-hf-username-codereview-env.hf.space") as client:

        # Start episode
        obs = await client.reset()
        print(obs.task_description)
        print(obs.code_snippet)

        # Submit a review
        action = ReviewAction(
            comments=[
                ReviewComment(
                    line_number=7,
                    issue_type="bug",
                    severity="high",
                    description="Loop starts at index 1 instead of 0, skipping the first element in the sum.",
                    suggested_fix="Change range(1, len(numbers)) to range(len(numbers))"
                )
            ],
            summary="Found one off-by-one bug in compute_average."
        )
        result = await client.step(action)
        print(f"Reward: {result.reward}")
        print(f"Feedback: {result.observation.feedback}")

asyncio.run(main())
```

---

### Run the baseline

```bash
export OPENAI_API_KEY=your_key_here

# Human-readable table
python codereview_env/baseline/run_baseline.py

# JSON output (used by /baseline endpoint)
python codereview_env/baseline/run_baseline.py --output-json
```

Expected baseline output (gpt-4o-mini):

```
Baseline model: gpt-4o-mini
Running on 3 tasks...

Task                 Difficulty      Score   Found/Total    FP
--------------------------------------------------------------
task_1_bug           easy           0.8750     1/1          0.00
task_2_security      medium         0.6250     2/3          0.00
task_3_multi         hard           0.4625     3/5          0.25
--------------------------------------------------------------
Average score                       0.6542
```

---

## Project Structure

```
codereview_env/
├── __init__.py                  # Package exports
├── models.py                    # Pydantic models: ReviewAction, CodeObservation, ReviewState, GraderResult
├── tasks.py                     # Task dataset with ground-truth issues
├── grader.py                    # Deterministic grader (0.0 – 1.0)
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Package metadata and dependencies
├── README.md                    # This file
├── server/
│   ├── __init__.py
│   ├── environment.py           # CodeReviewEnvironment(Environment) — reset/step/state
│   ├── app.py                   # FastAPI app with all endpoints
│   ├── requirements.txt         # Server dependencies for Docker
│   └── Dockerfile               # Container definition
└── baseline/
    ├── __init__.py
    └── run_baseline.py          # Baseline inference script (OpenAI-compatible)
```

---

## Disqualification Checklist

Before submitting, verify all pass:

- [ ] `docker build -f server/Dockerfile -t codereview-env . && docker run -p 7860:7860 codereview-env` — succeeds
- [ ] `curl http://localhost:7860/health` — returns `{"status": "ok"}`
- [ ] `curl http://localhost:7860/tasks` — returns all 3 tasks
- [ ] `curl -X POST http://localhost:7860/reset` — returns a `CodeObservation`
- [ ] `python codereview_env/baseline/run_baseline.py --output-json` — completes without error
- [ ] All 3 graders return scores in `[0.0, 1.0]`
- [ ] HF Space is tagged with `openenv`

---

## License

MIT
