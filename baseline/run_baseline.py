"""
codereview_env/baseline/run_baseline.py

Reproducible baseline inference script.
Connects to Groq using the OpenAI Python client and evaluates "llama-3.1-8b-instant"
against the local CodeReview-Env interface.
"""
import json
import os
import requests
from openai import OpenAI

# Initialize the Groq-compatible client
# Expects GROQ_API_KEY to be set in environment variables
API_KEY = os.getenv("GROQ_API_KEY", "")
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Deterministic evaluation target
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are a Senior Security & Performance Software Engineer.
Your task is to review the provided Python code snippet.
Find all bugs, security vulnerabilities, or logic flaws.

You MUST respond ONLY with a raw JSON object string. Do not include markdown fences, conversational text, or explanations outside the JSON.

SCHEMA REQUIREMENT:
{
  "summary": "Brief overview of findings",
  "comments": [
    {
      "line_number": <an integer, 1-indexed>,
      "issue_type": "<bug | security | style | performance | logic>",
      "severity": "<low | medium | high | critical>",
      "description": "Detailed explanation of the issue"
    }
  ]
}

If there are no issues, return an empty array for "comments".
"""

def run_evaluation(base_url: str = "http://localhost:7860"):
    """
    Runs the baseline inference loop against the target OpenEnv instance.
    """
    print(f"\n[=>] Starting Baseline Evaluation using {MODEL}")
    print(f"[=>] Environment URL: {base_url}\n")

    try:
        reset_resp = requests.post(f"{base_url}/reset")
        reset_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[!] Failed to connect to environment at {base_url}. Is the server running?\n{e}")
        return

    obs = reset_resp.json()
    done = obs.get("done", False)
    
    total_reward = 0.0
    task_count = 0

    while not done:
        task_id = obs.get('task_id', 'unknown')
        code_snip = obs.get('code_snippet', '')
        desc = obs.get('task_description', '')
        
        print(f"\n--- Task #{task_count + 1}: {task_id} ---")
        
        # 1. Build context
        user_prompt = f"Task Instruction: {desc}\n\nCode to review:\n{code_snip}"
        
        # 2. Query model deterministically
        try:
            chat_completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Zero temperature for maximum deterministic behavior
            )
            raw_output = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"[!] Model API error: {e}")
            print("[!] Providing empty string to environment to continue episode.")
            raw_output = ""

        # 3. Submit Action to Environment
        # The environment handles malformed JSON parsing safely
        try:
            step_resp = requests.post(
                f"{base_url}/step",
                data=raw_output,  # Sending raw text directly as instructed
                headers={"Content-Type": "text/plain"}
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[!] Environment Step Error: {e}")
            break

        # 4. Extract next state
        reward = step_data.get("reward", 0.0)
        info = step_data.get("info", {})
        obs = step_data.get("observation", {})
        done = step_data.get("done", False)

        total_reward += reward
        task_count += 1

        print(f"Reward Issued: {reward:.4f}")
        print(f"Grader Info Dict: {json.dumps(info, indent=2)}")

    if task_count > 0:
        avg_score = total_reward / task_count
    else:
        avg_score = 0.0

    print(f"\n{'='*40}")
    print(f"BASELINE EVALUATION COMPLETE")
    print(f"Total Tasks:   {task_count}")
    print(f"Average Score: {avg_score:.4f} / 1.0000")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    if not API_KEY:
        print("[WARN] GROQ_API_KEY environment variable not set.")
        print("[WARN] Connection to Groq might fail unless default fallback is configured.\n")
    
    run_evaluation()
