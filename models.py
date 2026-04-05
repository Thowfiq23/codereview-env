from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class AgentAction(BaseModel):
    # THE GOD MODE TOOLS
    action_type: Literal["execute_command", "patch_file", "submit_review"]
    
    # Tool 1: execute_command
    command: Optional[str] = Field(None, description="The raw bash command to run (e.g., 'cat file.py' or 'pytest')")
    
    # Tool 2: patch_file
    target_file: Optional[str] = Field(None, description="The file to overwrite")
    new_content: Optional[str] = Field(None, description="The complete new code to write to the file")
    
    # Tool 3: submit_review
    summary: Optional[str] = Field(None, description="Summary of how you fixed the bug.")

class CodeObservation(BaseModel):
    task_id: str
    context: str
    available_files: List[str]
    action_result: str
    step_number: int
    done: bool
    reward: float

class ReviewState(BaseModel):
    episode_id: str
    task_id: str
    step_count: int
    current_task_index: int
    total_reward: float
    done: bool

class StepResponse(BaseModel):
    observation: CodeObservation
    reward: float
    done: bool
    info: Dict[str, Any]