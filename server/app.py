import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request

from models import CodeObservation, AgentAction, EpisodeReward, ReviewState
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
# Serialize all mutating env calls so concurrent evaluator requests
# cannot interleave reset/step and corrupt episode state.
_env_lock = asyncio.Lock()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    # Return "healthy" as required by openenv validate --url
    return {"status": "healthy"}


@app.get("/metadata")
async def get_metadata() -> Dict[str, Any]:
    """OpenEnv spec: returns environment name and description."""
    return {
        "name": "codereview-env",
        "description": (
            "SWE-bench style autonomous code review and patching POMDP sandbox. "
            "Agents fix real bugs in Python repos; pytest grades the fix."
        ),
        "version": "1.0.0",
        "author": "Thowfiq Rahman A",
    }


@app.get("/schema")
async def get_schema() -> Dict[str, Any]:
    """OpenEnv spec: returns JSON schemas for action, observation, and state."""
    return {
        "action": AgentAction.model_json_schema(),
        "observation": CodeObservation.model_json_schema(),
        "state": ReviewState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> Dict[str, Any]:
    """OpenEnv spec: minimal JSON-RPC 2.0 MCP endpoint."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    method = body.get("method", "")
    req_id = body.get("id", 1)

    if method == "tools/list":
        tools = [
            {
                "name": "execute_command",
                "description": "Run a bash command in the sandbox.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action_type": {"type": "string", "const": "execute_command"},
                        "command": {"type": "string"},
                    },
                    "required": ["action_type", "command"],
                },
            },
            {
                "name": "patch_file",
                "description": "Overwrite a file with new content.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action_type": {"type": "string", "const": "patch_file"},
                        "target_file": {"type": "string"},
                        "new_content": {"type": "string"},
                    },
                    "required": ["action_type", "target_file", "new_content"],
                },
            },
            {
                "name": "submit_review",
                "description": "Submit the final review and trigger grading.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action_type": {"type": "string", "const": "submit_review"},
                        "summary": {"type": "string"},
                    },
                    "required": ["action_type", "summary"],
                },
            },
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    # Fallback: acknowledge unknown methods
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"status": "ok", "method": method},
    }


@app.get("/tasks")
async def get_tasks() -> Dict[str, list]:
    return {"tasks": list_task_ids()}


@app.post("/reset", response_model=CodeObservation)
async def reset_env(request: Request) -> CodeObservation:
    """
    Start a new episode.  Accepts an optional JSON body::

        {"task_id": "task_3_pr"}

    When ``task_id`` is provided the environment loads that specific task
    instead of advancing the internal round-robin counter.  Omitting the
    body (or sending ``{}``) keeps the default cycling behaviour.
    """
    task_id = None
    try:
        body_bytes = await request.body()
        if body_bytes:
            body = json.loads(body_bytes)
            if isinstance(body, dict):
                task_id = body.get("task_id")
    except Exception:
        task_id = None
    async with _env_lock:
        return env.reset(task_id=task_id)


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

    async with _env_lock:
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
