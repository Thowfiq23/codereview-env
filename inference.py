import os
import json
import requests
import textwrap
from openai import OpenAI
from client import CodeReviewEnv
from models import AgentAction
from tasks import TASKS

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("HF_TOKEN")  # Using your Groq key stored here
ENV_URL = "http://localhost:7860"
MAX_STEPS = 10  
TEMPERATURE = 0.1

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an elite Senior Security Engineer reviewing a GitHub Pull Request.
    You must investigate the repository using the available tools before submitting your final review.
    
    You have 3 tools available. You must choose exactly ONE tool per turn by outputting strict JSON.
    
    TOOL 1: read_file
    Use this to read the contents of a specific file.
    {"action_type": "read_file", "target_file": "path/to/file.py"}
    
    TOOL 2: search_code
    Use this to search the repository for a keyword.
    {"action_type": "search_code", "search_query": "password"}
    
    TOOL 3: submit_review
    Use this ONLY when you are done investigating and ready to submit your findings.
    {
      "action_type": "submit_review",
      "review_comments": [
        {
          "file_path": "path/to/file.py",
          "line_number": 42,
          "issue_type": "security",
          "severity": "high",
          "description": "SQL Injection found here."
        }
      ],
      "summary": "Overall summary of the PR."
    }
    
    RULES:
    1. Output ONLY valid JSON. No markdown formatting, no backticks, no explanations.
    2. Start by reading the README or searching for keywords.
    3. You must use 'submit_review' to finish.
""")

def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def run_evaluation():
    env = CodeReviewEnv(base_url=ENV_URL)
    print(f"\n[=>] Starting PR Review Agent using {MODEL_NAME}")
    
    total_score = 0.0
    num_tasks = len(TASKS)
    
    obs = env.reset()
    
    for task_idx in range(num_tasks):
        task_id = obs.task_id
        print(f"\n[START] task={task_id} env=codereview_v2 model={MODEL_NAME}")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"PR Context: {obs.context}\nAvailable Files: {obs.available_files}\n\nWhat tool will you use first?"}
        ]
        
        done = False
        step = 0
        task_reward = 0.0
        
        while not done and step < MAX_STEPS:
            step += 1
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                )
                raw_action = completion.choices[0].message.content
                clean_action = extract_json(raw_action)
                messages.append({"role": "assistant", "content": clean_action})
            except Exception as e:
                print(f"LLM Error: {e}")
                break

            try:
                action_obj = AgentAction.model_validate_json(clean_action)
                obs_data, reward, done, info = env.step(action_obj)
                obs_result = obs_data.action_result
                if done:
                    task_reward = reward
                    obs = obs_data # Prepare observation for next task if any
                    
            except Exception as e:
                error_msg = f"Env Error: {str(e)}"
                print(f"[STEP] step={step} action=INVALID reward=0.00 done=false error={error_msg}")
                messages.append({"role": "user", "content": f"SYSTEM ERROR: Invalid JSON or Action. {str(e)}"})
                continue
            
            try:
                action_dict = json.loads(clean_action)
                action_type = action_dict.get("action_type", "unknown")
            except:
                action_type = "invalid_json"

            print(f"[STEP] step={step} action={action_type} reward={reward:.2f} done={str(done).lower()} error=null")
            
            if not done:
                messages.append({"role": "user", "content": f"Observation from environment:\n{obs_result}\n\nWhat is your next tool action?"})
            
        success = "true" if task_reward > 0 else "false"
        print(f"[END] task={task_id} success={success} steps={step} final_reward={task_reward:.2f}")
        total_score += task_reward

    avg_score = total_score / num_tasks
    print(f"\n========================================")
    print(f"BASELINE EVALUATION COMPLETE")
    print(f"Total Tasks:   {num_tasks}")
    print(f"Average Score: {avg_score:.4f} / 1.0000")
    print(f"========================================")

if __name__ == "__main__":
    run_evaluation()
