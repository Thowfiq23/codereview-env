"""
codereview_env/server/app.py

FastAPI routes for the OpenEnv interface.
"""
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..models import CodeObservation, StepResponse
from ..tasks import TASKS, list_task_ids
from .environment import CodeReviewEnvironment

app = FastAPI(
    title="CodeReview-Env",
    description="OpenEnv interface for Python PR reviews. Evaluated deterministically with AST and fuzzy matching.",
    version="1.0.0"
)

# Singleton environment for the lifetime of the server.
env = CodeReviewEnvironment()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health probe."""
    return {"status": "ok"}


@app.post("/reset", response_model=CodeObservation)
async def reset_env() -> CodeObservation:
    """Reset the environment to task 1 and return the initial observation."""
    return env.reset()


@app.post("/step", response_model=StepResponse)
async def step_env(request: Request) -> StepResponse:
    """
    Submit an action (review response) to the environment.
    We accept the raw Request body so FastAPI validation doesn't crash on 
    malformed markdown or bad JSON before our unbreakable parser can handle it.
    """
    # Extract the raw payload text regardless of headers/content-type
    body_bytes = await request.body()
    try:
        raw_text = body_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = ""

    return env.step(raw_text)


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """Get the current tracking state of the episode."""
    return env.state()


@app.get("/tasks")
async def get_tasks_info() -> Dict[str, Any]:
    """
    Returns the task registry and exactly what JSON schema the agent must output
    in order to be evaluated successfully.
    """
    action_schema = {
        "type": "object",
        "properties": {
            "comments": {
                "type": "array",
                "description": "List of review comments. Empty list if code looks good.",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_number": {"type": "integer", "minimum": 1},
                        "issue_type": {
                            "type": "string",
                            "enum": ["bug", "security", "style", "performance", "logic"]
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"]
                        },
                        "description": {"type": "string", "minLength": 10},
                        "suggested_fix": {"type": ["string", "null"]}
                    },
                    "required": ["line_number", "issue_type", "severity", "description"]
                }
            },
            "summary": {"type": "string", "description": "Overall summary of review findings."}
        },
        "required": ["comments"]
    }

    return {
        "task_ids": list_task_ids(),
        "total_tasks": len(TASKS),
        "action_schema": action_schema
    }
