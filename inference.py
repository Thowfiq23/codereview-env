import os
import json
from openai import OpenAI
from client import CodeReviewEnv
from models import AgentAction

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("HF_TOKEN")
ENV_URL = "http://localhost:7860"
MAX_STEPS = 10  
TEMPERATURE = 0.1

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------
# THE GOD MODE ENGINEER (V4 SYSTEM PROMPT)
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are an elite Autonomous Software Engineer running inside a real Linux Sandbox.
You must fix the failing tests in the repository and prove your code works.

You have 3 tools available. You must choose exactly ONE tool per turn by outputting strict JSON.

TOOL 1: execute_command (Raw Bash Terminal)
Use this to run ANY linux command. (e.g., 'pytest tests/', 'cat auth/models.py', 'ls -la')
{"action_type": "execute_command", "command": "pytest tests/"}

TOOL 2: patch_file (Physical Disk Write)
Overwrite a file with entirely new content to fix bugs. You MUST provide the complete file content.
{"action_type": "patch_file", "target_file": "auth/models.py", "new_content": "def get_user_query():\n    return 'safe_query'"}

TOOL 3: submit_review
Use this ONLY when you have run 'pytest tests/' and verified that all tests are passing!
{"action_type": "submit_review", "summary": "Fixed SQL injection."}

RULES:
1. Output ONLY valid JSON. No markdown formatting.
2. ALWAYS run 'pytest tests/' first to see what is broken!
3. To read a file, use: {"action_type": "execute_command", "command": "cat filename.py"}
"""

def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def run_agent():
    env = CodeReviewEnv(base_url=ENV_URL)
    
    total_score = 0.0
    episodes = 3
    
    for i in range(episodes):
        obs = env.reset()
        print(f"\n==============================================")
        print(f"🚀 STARTING TASK: {obs.task_id}")
        print(f"==============================================")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {obs.context}\nAvailable files: {obs.available_files}\nAction Result: {obs.action_result}"}
        ]
        
        done = False
        step = 0
        episode_reward = 0.0
        
        while not done and step < MAX_STEPS:
            step += 1
            print(f"\n--- Step {step} ---")
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=2048
                )
                
                raw_action = response.choices[0].message.content
                print(f"🤖 LLM Raw Output:\n{raw_action}")
                
                clean_json = extract_json(raw_action)
                
                messages.append({"role": "assistant", "content": clean_json})
                
                action_obj = AgentAction.model_validate_json(clean_json)
                obs, reward, done, info = env.step(action_obj)
                episode_reward += reward
                
                print(f"🌍 Env Output:\n{obs.action_result}")
                
                messages.append({"role": "user", "content": f"Action Result:\n{obs.action_result}"})
                
                if done:
                    print(f"\n🎯 Task {obs.task_id} Complete!")
                    print(f"🏆 Final Reward for this task: {reward}")
                    print(f"📊 Feedback: {info.get('feedback', 'None')}")
                    
            except Exception as e:
                print(f"⚠️ Error during inference step {step}: {e}")
                break
                
        total_score += episode_reward

    print("\n" + "="*50)
    print(f"🏁 ALL TASKS COMPLETE")
    print(f"🌟 AVERAGE SCORE: {total_score / episodes:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_agent()
