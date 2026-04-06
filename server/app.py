import json
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request

from models import CodeObservation, AgentAction, EpisodeReward
from tasks import TASKS, list_task_ids
from .environment import CodeReviewEnvironment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean up workspace on server shutdown."""
    yield
    env.cleanup()


app = FastAPI(
    title="CodeReview-Env V4",
    description="Autonomous SWE sandbox — agents fix real bugs, pytest grades the fix.",
    version="4.0.0",
    lifespan=lifespan,
)

env = CodeReviewEnvironment()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
async def get_tasks() -> Dict[str, list]:
    return {"tasks": list_task_ids()}


@app.post("/reset", response_model=CodeObservation)
async def reset_env() -> CodeObservation:
    return env.reset()


@app.post("/step")
async def step_env(request: Request) -> Dict[str, Any]:
    raw_text = (await request.body()).decode("utf-8")

    try:
        action_dict = json.loads(raw_text)
        action_obj = AgentAction.model_validate(action_dict)
    except Exception as e:
        # Fall back to a safe no-op so the episode is not silently broken
        action_obj = AgentAction(
            action_type="execute_command",
            command=f"echo 'Parse error — send valid JSON: {str(e)[:120]}'"
        )

    obs, reward, done, info = env.step(action_obj)

    # Include typed EpisodeReward in the response to satisfy OpenEnv spec
    typed_reward = EpisodeReward(
        value=round(float(reward), 4),
        is_terminal=bool(done),
        breakdown={"test_pass_rate": round(float(reward), 4)},
    )

    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
        "typed_reward": typed_reward.model_dump(),
    }


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    return env.state.model_dump()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
