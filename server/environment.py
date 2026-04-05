import os
import uuid
import subprocess
import shutil
from typing import Dict, Any, Tuple
from models import AgentAction, CodeObservation, ReviewState
from tasks import TASKS

WORKSPACE_BASE = "/tmp/codereview_workspaces"

class CodeReviewEnvironment:
    def __init__(self):
        self.state = ReviewState(
            episode_id=str(uuid.uuid4()),
            task_id="",
            step_count=0,
            current_task_index=0,
            total_reward=0.0,
            done=False
        )
        self.current_task_data = None
        self._is_first_reset = True
        self.workspace_dir = ""

    def _setup_workspace(self):
        """Physically writes the task files to the server's hard drive."""
        self.workspace_dir = os.path.join(WORKSPACE_BASE, self.state.episode_id)
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        repo = self.current_task_data["repository"]
        for filepath, content in repo.items():
            full_path = os.path.join(self.workspace_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)

    def reset(self) -> CodeObservation:
        self.state.episode_id = str(uuid.uuid4())
        self.state.step_count = 0
        self.state.done = False
        self.state.total_reward = 0.0
        
        if not self._is_first_reset:
            self.state.current_task_index = (self.state.current_task_index + 1) % len(TASKS)
        self._is_first_reset = False
        
        self.current_task_data = TASKS[self.state.current_task_index]
        self.state.task_id = self.current_task_data["task_id"]
        
        # Deploy the physical sandbox!
        self._setup_workspace()
        
        available_files = list(self.current_task_data["repository"].keys())
        
        return CodeObservation(
            task_id=self.state.task_id,
            context=self.current_task_data["description"],
            available_files=available_files,
            action_result=f"Sandbox deployed at {self.workspace_dir}. You have a real bash terminal. Run 'pytest tests/' to begin.",
            step_number=self.state.step_count,
            done=False,
            reward=0.0
        )

    def step(self, action_obj: AgentAction) -> Tuple[CodeObservation, float, bool, Dict[str, Any]]:
        try:
            if self.state.done:
                raise ValueError("Episode is already done. Call reset().")
                
            self.state.step_count += 1
            available_files = list(self.current_task_data["repository"].keys())
            
            obs_result = ""
            reward = 0.0
            done = False
            info = {}

            # --- TOOL 1: EXECUTE COMMAND (Real Linux Bash) ---
            if action_obj.action_type == "execute_command":
                cmd = action_obj.command
                if not cmd:
                    obs_result = "Error: No bash command provided."
                else:
                    # Run the command physically in the workspace directory
                    result = subprocess.run(
                        cmd, cwd=self.workspace_dir, shell=True, 
                        capture_output=True, text=True, timeout=15
                    )
                    output = result.stdout if result.stdout else result.stderr
                    obs_result = f"--- BASH OUTPUT (Exit Code {result.returncode}) ---\n{output}"

            # --- TOOL 2: PATCH FILE (Physical Disk Write) ---
            elif action_obj.action_type == "patch_file":
                target = action_obj.target_file
                new_content = action_obj.new_content
                if not target or not new_content:
                    obs_result = "Error: target_file and new_content required."
                else:
                    full_path = os.path.join(self.workspace_dir, target)
                    if ".." in target:
                        obs_result = "Security Error: Path traversal blocked."
                    else:
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, "w") as f:
                            f.write(new_content)
                        obs_result = f"Successfully overwrote physical file: {target}"

            # --- TOOL 3: SUBMIT REVIEW (The Execution Grader) ---
            elif action_obj.action_type == "submit_review":
                # The ultimate test: We run pytest on their code!
                result = subprocess.run(
                    "pytest tests/", cwd=self.workspace_dir, shell=True, 
                    capture_output=True, text=True, timeout=15
                )
                
                if result.returncode == 0:
                    reward = 1.0  # PERFECT SCORE! They actually fixed the app.
                    feedback = "SUCCESS! All tests passed in the sandbox."
                else:
                    reward = 0.0  # Tests still failing.
                    feedback = f"FAILED! Tests are still failing.\n{result.stdout[-500:]}"
                    
                obs_result = f"Evaluation complete. {feedback}"
                done = True
                self.state.total_reward += reward

            obs = CodeObservation(
                task_id=self.state.task_id, context=self.current_task_data["description"],
                available_files=available_files, action_result=obs_result,
                step_number=self.state.step_count, done=done, reward=reward
            )
            self.state.done = done
            return obs, reward, done, info
            
        except Exception as e:
            obs = CodeObservation(
                task_id=self.state.task_id if self.current_task_data else "error",
                context="Error state", available_files=[], action_result=f"Internal Env Error: {str(e)}",
                step_number=self.state.step_count, done=False, reward=0.0
            )
            return obs, 0.0, False, {"error": str(e)}
