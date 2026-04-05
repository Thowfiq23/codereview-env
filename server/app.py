import json
from typing import Any, Dict

from fastapi import FastAPI, Request

from models import CodeObservation, AgentAction
from tasks import TASKS, list_task_ids
from .environment import CodeReviewEnvironment

app = FastAPI(
    title="CodeReview-Env V4 (God Mode)",
    description="Physical execution sandbox with subprocess and pytest.",
    version="4.0.0"
)

env = CodeReviewEnvironment()

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/reset", response_model=CodeObservation)
async def reset_env() -> CodeObservation:
    return env.reset()

@app.post("/step")
async def step_env(request: Request):
    raw_text = (await request.body()).decode("utf-8")
    
    try:
        # Parse the incoming JSON into our strict Pydantic God Mode Action
        action_dict = json.loads(raw_text)
        action_obj = AgentAction.model_validate(action_dict)
    except Exception as e:
        # Fallback if the AI sends total garbage JSON
        action_obj = AgentAction(action_type="execute_command", command=f"echo 'Invalid JSON: {e}'")
        
    # Pass the validated object to the physical sandbox
    obs, reward, done, info = env.step(action_obj)
    
    return {
        "observation": obs.model_dump(), 
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/state")
async def get_state() -> Dict[str, Any]:
    return env.state.model_dump()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()