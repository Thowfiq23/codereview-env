"""
codereview_env/models.py
Typed Pydantic models for the CodeReview OpenEnv environment.
All models follow the OpenEnv spec: Action, Observation, State.
"""
from __future__ import annotations
from typing import Dict, List, Literal, Optional, Any
from uuid import uuid4
from pydantic import BaseModel, Field

class ReviewComment(BaseModel):
    line_number: int = Field(..., ge=1)
    issue_type: Literal["bug", "security", "style", "performance", "logic"]
    severity: Literal["low", "medium", "high", "critical"]
    description: str = Field(..., min_length=10)
    suggested_fix: Optional[str] = None

class ReviewAction(BaseModel):
    comments: List[ReviewComment] = Field(default_factory=list)
    summary: str = ""

class CodeObservation(BaseModel):
    task_id: str
    task_description: str
    code_snippet: str
    filename: str
    language: str = "python"
    context: str = ""
    step_number: int = 0
    feedback: Optional[str] = None
    done: bool = False
    reward: float = 0.0

class StepResponse(BaseModel):
    observation: CodeObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class ReviewState(BaseModel):
    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = ""
    step_count: int = 0
    current_task_index: int = 0
    total_reward: float = 0.0
    done: bool = False

class GraderResult(BaseModel):
    task_id: str
    total_score: float = Field(..., ge=0.0, le=1.0)
    issues_found: int
    total_issues: int
    recall_score: float
    severity_accuracy: float
    false_positive_penalty: float
    per_issue_detail: List[Dict[str, Any]] = Field(default_factory=list)
    feedback: str = ""
