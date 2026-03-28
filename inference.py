"""
codereview_env/inference.py

Reproducible baseline inference script.
Connects to an LLM using the OpenAI Python client and evaluates it
against the local CodeReview-Env interface.
"""
import json
import os
import requests
from openai import OpenAI

# Hackathon Required Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
# Fallbacks included so your local testing still works
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", "")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

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

def run_evaluation():
    """
    Runs the baseline inference directly against the codebase logic.
    """
    print(f"\n[=>] Starting Baseline Evaluation using {MODEL}")

    done = False
    total_reward = 0.0
    task_count = 0

    # Start by getting the first task from tasks.py directly without HTTP
    from tasks import TASKS
    
    for task in TASKS:
        task_id = task.get('id', 'unknown')
        code_snip = task.get('code_snippet', '')
        desc = task.get('task_description', '')
        
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
                temperature=0.0,
            )
            raw_output = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"[!] Model API error: {e}")
            print("[!] Providing empty string to environment to continue episode.")
            raw_output = ""

        # 3. Grade the result directly
        from grader import parse_agent_action, evaluate_review
        
        action = parse_agent_action(raw_output)
        if action is None:
            reward = 0.0
            info = {"error": "Invalid action format."}
        else:
            reward, info = evaluate_review(
                agent_action=action,
                code_snippet=code_snip,
                ground_truth_issues=task["ground_truth_issues"]
            )

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

    return {
        "average_score": avg_score,
        "tasks_evaluated": task_count,
        "model_used": MODEL
    }

if __name__ == "__main__":
    if not API_KEY:
        print("[WARN] OPENAI_API_KEY or GROQ_API_KEY environment variable not set.")
        print("[WARN] Connection to API might fail unless default fallback is configured.\n")
    
    run_evaluation()
