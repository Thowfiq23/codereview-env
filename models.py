"""
codereview_env/models.py
Typed Pydantic models for the CodeReview OpenEnv environment.
All models follow the OpenEnv spec: Action, Observation, State.
"""
from __future__ import annotations
from typing import Dict, List, Literal, Optional, Any
from uuid import uuid4
from pydantic import BaseModel, Field

# ── Action ────────────────────────────────────────────────────────────────────

class ReviewComment(BaseModel):
    file_path: str = Field(..., description="The exact path of the file being commented on.")
    line_number: int = Field(..., ge=1)
    issue_type: Literal["bug", "security", "style", "performance", "logic"]
    severity: Literal["low", "medium", "high", "critical"]
    description: str = Field(..., min_length=10)
    suggested_fix: Optional[str] = None

class AgentAction(BaseModel):
    """
    The unified action space. The agent must choose ONE action_type per step.
    If action_type is 'read_file', it must provide 'target_file'.
    If action_type is 'search_code', it must provide 'search_query'.
    If action_type is 'submit_review', it must provide 'review_comments' and 'summary'.
    """
    action_type: Literal["read_file", "search_code", "submit_review"] = Field(
        ..., description="The tool you want to use for this step."
    )
    
    # Tool 1 Parameters
    target_file: Optional[str] = Field(None, description="Used with 'read_file'. The path of the file to open.")
    
    # Tool 2 Parameters
    search_query: Optional[str] = Field(None, description="Used with 'search_code'. The exact string to search for across the repo.")
    
    # Tool 3 Parameters (The Final Action)
    review_comments: Optional[List[ReviewComment]] = Field(None, description="Used with 'submit_review'. List of findings.")
    summary: Optional[str] = Field(None, description="Used with 'submit_review'. Overall PR summary.")

# ── Observation ───────────────────────────────────────────────────────────────

class CodeObservation(BaseModel):
    task_id: str
    context: str = Field(..., description="The PR description or current state context.")
    available_files: List[str] = Field(..., description="Files modified in this PR.")
    
    # The result of the agent's last action (e.g., file contents, search results, or error messages)
    action_result: Optional[str] = None 
    
    step_number: int
    feedback: Optional[str] = None
    done: bool
    reward: float

# ── Environment Interop ───────────────────────────────────────────────────────

class StepResponse(BaseModel):
    """Standard OpenEnv step response."""
    observation: CodeObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

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
    per_issue_detail: List[Dict[str, Any]] = Field(default_factory=list)
    feedback: str = Field(default="", description="Human-readable grader feedback.")

