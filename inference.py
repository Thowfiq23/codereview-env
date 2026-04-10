"""
CodeReview-Env Inference Script
================================
Runs an LLM agent against all 3 tasks in the CodeReview-Env sandbox.

Required env vars:
  API_BASE_URL    - LLM API endpoint (default: HF Router)
  MODEL_NAME      - Model identifier
  HF_TOKEN        - Hugging Face API key (mandatory, no default)
  ENV_URL         - Environment server URL (default: http://localhost:7860)

Stdout format (mandatory — exact OpenEnv spec):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import os
import sys
import json
import logging
import time

# Suppress httpx/httpcore/openai INFO logs — they can emit to stdout and
# break the strict [START]/[STEP]/[END] format the evaluator regex-parses.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Route any uncaught exception (import error, syntax error, etc.) to stderr
# so stdout stays clean for the evaluator.
def _global_excepthook(exctype, value, tb):
    import traceback
    print(f"[FATAL] {exctype.__name__}: {value}", file=sys.stderr, flush=True)
    traceback.print_tb(tb, file=sys.stderr)
sys.excepthook = _global_excepthook

from openai import OpenAI
from client import CodeReviewEnv
from models import AgentAction

# ---------------------------------------------------------------------------
# CONFIGURATION — all overridable via environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
API_KEY      = HF_TOKEN
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK        = "codereview-env"
MAX_STEPS        = 10
TEMPERATURE      = 0.0
# Upper bound for score normalisation.  A perfect efficient episode (one
# explore + two patch steps + submit) sums to roughly 1.9 in rewards.
# Using 2.0 keeps scores in a natural range and penalises inefficient
# trajectories (many wasted steps) relative to efficient ones.
MAX_TOTAL_REWARD = 2.0

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an elite Autonomous Software Engineer running inside a real Linux Sandbox.
You must fix the failing tests in the repository and prove your code works.

You have 3 tools. Choose exactly ONE per turn by outputting strict JSON only.

TOOL 1: execute_command — run any bash command
{"action_type": "execute_command", "command": "pytest tests/"}

TOOL 2: patch_file — overwrite a file completely to fix bugs
{"action_type": "patch_file", "target_file": "auth/models.py", "new_content": "def get_user_query(username):\\n    return ('SELECT * FROM users WHERE username = %s', (username,))"}

TOOL 3: submit_review — submit ONLY after pytest tests/ passes
{"action_type": "submit_review", "summary": "Fixed SQL injection and MD5 hashing."}

RULES:
1. Output ONLY valid JSON. No markdown, no backticks, no explanation.
2. Start EVERY episode by running: {"action_type": "execute_command", "command": "pytest tests/"}
3. Read a file with: {"action_type": "execute_command", "command": "cat path/to/file.py"}
4. Only call submit_review when all tests pass.
"""


def extract_json(text: str) -> str:
    """Strip markdown fences that LLMs sometimes wrap around JSON."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def run_task(env: CodeReviewEnv, task_index: int) -> float:
    """Run one full episode. Always emits [START]/[STEP]/[END] lines to stdout."""
    rewards = []
    step = 0
    score = 0.0
    task_name = f"task_{task_index + 1}_pr"

    try:
        obs = env.reset()
        task_name = obs.task_id
    except Exception as exc:
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        print(f"[STEP] step=1 action=null reward=0.01 done=false error=env_reset_failed:{exc}", flush=True)
        print(f"[END] success=false steps=1 score=0.01 rewards=0.01", flush=True)
        return 0.01

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {obs.context}\n"
                f"Available files: {obs.available_files}\n"
                "Start by running pytest tests/ to see what is broken."
            )
        }
    ]

    done = False

    try:
        while not done and step < MAX_STEPS:
            step += 1
            action_str = "null"
            error_str = "null"
            reward = 0.01  # floor: API/model errors must not log exactly 0.0

            # Track whether an assistant turn was committed so we always
            # close it with a user turn — even on validation/step failure.
            assistant_appended = False

            try:
                # Retry up to 3 times on rate-limit / quota errors (HTTP 429)
                # with exponential backoff so the episode isn't wasted on a
                # transient API throttle.
                _last_api_exc = None
                response = None
                for _attempt in range(3):
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            temperature=TEMPERATURE,
                            max_tokens=2048,
                            stream=False,
                        )
                        break
                    except Exception as _api_exc:
                        _last_api_exc = _api_exc
                        _msg = str(_api_exc).lower()
                        # Only retry on rate-limit / quota / server-overload signals
                        if any(tok in _msg for tok in ("429", "rate limit", "quota", "too many", "overloaded")):
                            if _attempt < 2:
                                time.sleep(2 ** _attempt)  # 1s, 2s
                                continue
                        raise  # Non-retryable — propagate immediately
                if response is None:
                    raise _last_api_exc

                raw = (response.choices[0].message.content or "").strip()
                clean = extract_json(raw)

                # Compact single-line representation for the [STEP] log
                try:
                    action_str = json.dumps(json.loads(clean), separators=(",", ":"))
                except Exception:
                    action_str = clean.replace("\n", " ").replace("\r", "")[:300]

                messages.append({"role": "assistant", "content": clean})
                assistant_appended = True

                action_obj = AgentAction.model_validate_json(clean)
                obs, reward, done, info = env.step(action_obj)

                # error= must reflect the env-side action error, not a local exception
                env_error = info.get("error") if info else None
                if env_error:
                    error_str = str(env_error).replace("\n", " ")[:200]

                messages.append({
                    "role": "user",
                    "content": f"Action Result:\n{obs.action_result}"
                })

            except Exception as exc:
                error_str = str(exc).replace("\n", " ")[:200]
                # If an assistant turn was committed but no user reply followed,
                # inject an error message so the LLM sees its mistake and can
                # self-correct on the next step instead of repeating the same
                # malformed output until MAX_STEPS is exhausted.
                if assistant_appended:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"ERROR: Your response could not be executed: {error_str}\n"
                            "Output ONLY valid JSON with no markdown, no backticks, "
                            "and no explanation. Try again."
                        )
                    })

            rewards.append(reward)
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.3f} done={str(done).lower()} error={error_str}",
                flush=True
            )

    except Exception as fatal:
        # Log catastrophic failures to stderr so the evaluator has diagnostics.
        # Stdout must stay clean ([START]/[STEP]/[END] only).
        print(f"[FATAL] task={task_name} error={str(fatal)[:300]}",
              file=__import__('sys').stderr, flush=True)

    # Score = sum(rewards) / MAX_TOTAL_REWARD, matching the spec sample formula.
    # This penalises inefficient trajectories (many wasted steps, destructive
    # actions) relative to clean, direct solves.  Clamped to (0.01, 0.99) so
    # the evaluator always sees a score in the open interval (0, 1).
    raw_score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0
    score = min(max(raw_score, 0.01), 0.99)
    success = score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # [END] format follows the OpenEnv spec (sample inference.py):
    # success steps score rewards  — evaluator regex-parses score= for task scoring
    print(
        f"[END] success={str(success).lower()} steps={step} score={score:.3f} rewards={rewards_str}",
        flush=True
    )

    return score


def main():
    env = CodeReviewEnv(base_url=ENV_URL)
    scores = []
    for i in range(3):
        scores.append(run_task(env, i))

    avg = sum(scores) / len(scores) if scores else 0.0
    # Print summary to stderr — stdout must contain only [START]/[STEP]/[END] per spec
    print(
        f"[SUMMARY] tasks=3 scores={','.join(f'{s:.2f}' for s in scores)} "
        f"average={avg:.2f}",
        file=__import__('sys').stderr, flush=True
    )


if __name__ == "__main__":
    main()
