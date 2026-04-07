"""
CodeReview-Env Inference Script
================================
Runs an LLM agent against all 3 tasks in the CodeReview-Env sandbox.

Required env vars:
  API_BASE_URL  - LLM API endpoint (default: HF Router)
  MODEL_NAME    - Model identifier
  HF_TOKEN      - API key
  ENV_URL       - Environment server URL (default: http://localhost:7860)

Stdout format (mandatory):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import os
import json
from openai import OpenAI
from client import CodeReviewEnv
from models import AgentAction

# ---------------------------------------------------------------------------
# CONFIGURATION — all overridable via environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "codereview-env"
MAX_STEPS    = 10
TEMPERATURE  = 0.0

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
        print(f"[STEP] step=1 action=null reward=0.00 done=false error=env_reset_failed:{exc}", flush=True)
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00", flush=True)
        return 0.0

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
            reward = 0.0

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=2048,
                    stream=False,
                )
                raw = (response.choices[0].message.content or "").strip()
                clean = extract_json(raw)

                # Compact single-line representation for the [STEP] log
                try:
                    action_str = json.dumps(json.loads(clean), separators=(",", ":"))
                except Exception:
                    action_str = clean.replace("\n", " ").replace("\r", "")[:300]

                messages.append({"role": "assistant", "content": clean})

                action_obj = AgentAction.model_validate_json(clean)
                obs, reward, done, info = env.step(action_obj)

                messages.append({
                    "role": "user",
                    "content": f"Action Result:\n{obs.action_result}"
                })

            except Exception as exc:
                error_str = str(exc).replace("\n", " ")[:200]

            rewards.append(reward)
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
                flush=True
            )

    except Exception:
        pass

    # Score = highest reward achieved during the episode (clamped to [0, 1])
    score = min(max(max(rewards) if rewards else 0.0, 0.0), 1.0)
    success = score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={score:.2f} rewards={rewards_str}",
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
