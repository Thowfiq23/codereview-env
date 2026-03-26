"""
codereview_env/models.py
Typed Pydantic models for the CodeReview OpenEnv environment.
All models follow the OpenEnv spec: Action, Observation, State.
"""
from __future__ import annotations
from typing import Dict, List, Literal, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


# ── Action ────────────────────────────────────────────────────────────────────

class ReviewComment(BaseModel):
    """A single issue identified by the agent in the code under review."""
    line_number: int = Field(..., description="1-indexed line number where the issue was found.", ge=1)
    issue_type: Literal["bug", "security", "style", "performance", "logic"] = Field(..., description="Category of the issue.")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="Severity level.")
    description: str = Field(..., description="Human-readable explanation of the issue.", min_length=10)
    suggested_fix: Optional[str] = Field(default=None, description="Optional suggested correction.")


class ReviewAction(BaseModel):
    """
    The agent's complete review of the code snippet.
    Submit a list of ReviewComment objects, one per identified issue.
    Submit an empty list if no issues are found.
    """
    comments: List[ReviewComment] = Field(default_factory=list, description="List of review comments.")
    summary: str = Field(default="", description="Overall summary of review findings.")


# ── Observation ───────────────────────────────────────────────────────────────

class CodeObservation(BaseModel):
    """What the agent sees: a code snippet (PR diff) plus metadata."""
    task_id: str = Field(..., description="Unique ID of the current task.")
    task_description: str = Field(..., description="Instruction telling the agent what to look for.")
    code_snippet: str = Field(..., description="The Python code to review.")
    filename: str = Field(..., description="File name of the code being reviewed.")
    language: str = Field(default="python", description="Programming language.")
    context: str = Field(default="", description="Background: what the code is supposed to do.")
    step_number: int = Field(default=0, description="Current step within the episode.")
    feedback: Optional[str] = Field(default=None, description="Grader feedback from the previous step.")
    done: bool = Field(default=False, description="Whether the episode has ended.")
    reward: float = Field(default=0.0, description="Reward from the previous step.")


# ── State ─────────────────────────────────────────────────────────────────────

class ReviewState(BaseModel):
    """Internal episode state tracked by the environment server."""
    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = Field(default="")
    step_count: int = Field(default=0)
    current_task_index: int = Field(default=0)
    total_reward: float = Field(default=0.0)
    done: bool = Field(default=False)


# ── Grader result (returned by /grader endpoint) ──────────────────────────────

class GraderResult(BaseModel):
    """Detailed reward breakdown for one agent ReviewAction on one task."""
    task_id: str
    total_score: float = Field(..., ge=0.0, le=1.0)
    issues_found: int
    total_issues: int
    recall_score: float = Field(..., description="Fraction of real issues found (0-1).")
    severity_accuracy: float = Field(..., description="Correct severity labels (0-1).")
    false_positive_penalty: float = Field(..., description="Penalty for hallucinated issues.")
    per_issue_detail: List[Dict] = Field(default_factory=list)
    feedback: str = Field(default="", description="Human-readable grader feedback.")
