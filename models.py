from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class AgentAction(BaseModel):
    """Action submitted by the agent each turn."""
    action_type: Literal["execute_command", "patch_file", "submit_review"]

    # Tool 1: execute_command
    command: Optional[str] = Field(
        None,
        description="Bash command to run in the sandbox (e.g. 'pytest tests/' or 'cat auth/models.py')"
    )

    # Tool 2: patch_file
    target_file: Optional[str] = Field(
        None,
        description="Relative path of the file to overwrite (e.g. 'auth/models.py')"
    )
    new_content: Optional[str] = Field(
        None,
        description="Complete new content to write to target_file"
    )

    # Tool 3: submit_review
    summary: Optional[str] = Field(
        None,
        description="Human-readable summary of the fix. Triggers final pytest grading."
    )


class CodeObservation(BaseModel):
    """Observation returned by reset() and step()."""
    task_id:         str        = Field(..., description="Task identifier, e.g. 'task_1_pr'")
    context:         str        = Field(..., description="PR description shown to the agent")
    available_files: List[str]  = Field(..., description="Files present in the sandbox")
    action_result:   str        = Field(..., description="Stdout/stderr of the last command, or a status message")
    step_number:     int        = Field(..., description="Number of steps taken so far in this episode")
    done:            bool       = Field(..., description="True when the episode has ended")
    reward:          float      = Field(..., description="Reward earned this step [0.0, 1.0]")


class EpisodeReward(BaseModel):
    """Typed reward model as required by OpenEnv spec."""
    value:       float            = Field(..., ge=0.0, le=1.0, description="Scalar reward [0.0, 1.0]")
    is_terminal: bool             = Field(False,              description="True if this reward ends the episode")
    breakdown:   Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component breakdown, e.g. {'test_pass_rate': 0.5}"
    )


class ReviewState(BaseModel):
    """Full episode state returned by state()."""
    episode_id:          str   = Field(..., description="UUID of the current episode")
    task_id:             str   = Field(..., description="Task currently active")
    step_count:          int   = Field(..., description="Steps taken in this episode")
    current_task_index:  int   = Field(..., description="Index into the TASKS list (cycles 0→1→2→0)")
    total_reward:        float = Field(0.0, description="Highest reward achieved so far this episode")
    done:                bool  = Field(..., description="True when the episode has ended")


class StepResponse(BaseModel):
    """Full response envelope from POST /step."""
    observation:  CodeObservation
    reward:       float
    done:         bool
    info:         Dict[str, Any]
    typed_reward: EpisodeReward
