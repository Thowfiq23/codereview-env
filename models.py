from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class AgentAction(BaseModel):
    """Action submitted by the agent each turn."""
    action_type: Literal[
        # Exploration tools
        "execute_command",   # escape-hatch: run any bash command
        "read_file",         # clean file reader — no shell needed
        "search_codebase",   # grep with a regex pattern across all .py files
        # Execution tools
        "patch_file",        # overwrite a source file with fixed content
        "run_tests",         # targeted pytest (optional path via `path` field)
        "restart_service",   # run app_init.py and update service_state.json
        # Diagnostic tools
        "inspect_logs",      # read service.log / migration.log from workspace
        # Terminal actions
        "submit_fix",        # preferred terminal: grade + record root_cause
        "submit_review",     # backward-compat alias for submit_fix
    ]

    # execute_command
    command: Optional[str] = Field(
        None,
        description="Bash command to run in the sandbox (e.g. 'ls -la' or 'python -c ...')"
    )

    # read_file / run_tests
    path: Optional[str] = Field(
        None,
        description="Relative file path for read_file (e.g. 'app/handler.py') "
                    "or test path for run_tests (e.g. 'tests/test_service.py'). "
                    "Omit path for run_tests to run the full test suite."
    )

    # search_codebase
    pattern: Optional[str] = Field(
        None,
        description="Regex pattern to search for across all .py files (e.g. 'TIMEOUT|MAX_WORKERS')"
    )

    # patch_file
    target_file: Optional[str] = Field(
        None,
        description="Relative path of the file to overwrite (e.g. 'config/settings.py')"
    )
    new_content: Optional[str] = Field(
        None,
        description="Complete new content to write to target_file"
    )

    # submit_fix / submit_review
    summary: Optional[str] = Field(
        None,
        description="Human-readable summary of the fix. Triggers final pytest grading."
    )
    root_cause: Optional[str] = Field(
        None,
        description="Root-cause analysis for submit_fix (e.g. 'TIMEOUT was str not int; "
                    "migration sort was lexicographic not numeric'). Recorded in episode metadata."
    )


class CodeObservation(BaseModel):
    """Observation returned by reset() and step()."""
    task_id:         str        = Field(..., description="Task identifier, e.g. 'task_1_pr'")
    context:         str        = Field(..., description="PR description shown to the agent")
    available_files: List[str]  = Field(..., description="Files present in the sandbox")
    action_result:   str        = Field(..., description="Stdout/stderr of the last command, or a status message")
    step_number:     int        = Field(..., description="Number of steps taken so far in this episode")
    done:            bool       = Field(..., description="True when the episode has ended")
    reward:          float      = Field(..., description="Reward earned this step. Range: [-0.1, 1.0]. Negative for destructive actions (path traversal, test tampering).")


class EpisodeReward(BaseModel):
    """Typed reward model as required by OpenEnv spec."""
    value:       float            = Field(..., description="Scalar reward in [-0.1, 1.0]. Negative values signal destructive actions.")
    is_terminal: bool             = Field(False, description="True if this reward ends the episode")
    breakdown:   Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component breakdown: test_pass_rate (0.0–1.0) and shaped_reward (the actual step reward)."
    )


class ReviewState(BaseModel):
    """Full episode state returned by state()."""
    episode_id:          str   = Field(..., description="UUID of the current episode")
    task_id:             str   = Field(..., description="Task currently active")
    step_count:          int   = Field(..., description="Steps taken in this episode")
    current_task_index:  int   = Field(..., description="Index into the TASKS list (cycles 0→1→…→5→0)")
    total_reward:        float = Field(0.0, description="Highest reward achieved so far this episode")
    done:                bool  = Field(..., description="True when the episode has ended")


class StepResponse(BaseModel):
    """Full response envelope from POST /step."""
    observation:  CodeObservation
    reward:       float
    done:         bool
    info:         Dict[str, Any]
    typed_reward: EpisodeReward
