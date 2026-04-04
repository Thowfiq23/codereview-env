import uuid
from typing import Dict, Any, Tuple
from models import AgentAction, CodeObservation, ReviewState
from tasks import TASKS
from grader import evaluate_review, parse_agent_action

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
        self._is_first_reset = True  # Safe cycler flag

    def reset(self) -> CodeObservation:
        self.state.episode_id = str(uuid.uuid4())
        self.state.step_count = 0
        self.state.done = False
        self.state.total_reward = 0.0
        
        # Cycle through tasks cleanly
        if not self._is_first_reset:
            self.state.current_task_index = (self.state.current_task_index + 1) % len(TASKS)
        self._is_first_reset = False
        
        # Load the task
        self.current_task_data = TASKS[self.state.current_task_index]
        self.state.task_id = self.current_task_data["task_id"]
        
        available_files = list(self.current_task_data["repository"].keys())
        
        return CodeObservation(
            task_id=self.state.task_id,
            context=self.current_task_data["description"],
            available_files=available_files,
            action_result="Environment reset. Please select a file to read or search the codebase.",
            step_number=self.state.step_count,
            done=False,
            reward=0.0
        )

    def step(self, action_text: str) -> Tuple[CodeObservation, float, bool, Dict[str, Any]]:
        try:
            if self.current_task_data is None:
                raise ValueError("Environment not initialized. You must call /reset first.")
            if self.state.done:
                raise ValueError("Episode is already done. Call reset().")
                
            self.state.step_count += 1
            repo = self.current_task_data["repository"]
            available_files = list(repo.keys())
            
            action = parse_agent_action(action_text)
            if not action:
                obs = CodeObservation(
                    task_id=self.state.task_id, context=self.current_task_data["description"],
                    available_files=available_files, action_result="Error: Invalid JSON format.",
                    step_number=self.state.step_count, done=False, reward=0.0
                )
                return obs, 0.0, False, {"error": "Invalid action schema."}

            obs_result = ""
            reward = 0.0
            done = False
            info = {}

            if action.action_type == "read_file":
                target = action.target_file
                obs_result = f"--- CONTENTS OF {target} ---\n{repo[target]}" if target in repo else f"Error: File '{target}' not found."

            elif action.action_type == "search_code":
                query = action.search_query or ""
                matches = [f"{fname} (Line {i+1}): {line.strip()}" for fname, content in repo.items() for i, line in enumerate(content.split('\n')) if query.lower() in line.lower()]
                obs_result = "--- SEARCH RESULTS ---\n" + "\n".join(matches) if matches else f"No matches found for '{query}'."

            elif action.action_type == "submit_review":
                issues = self.current_task_data["ground_truth_issues"]
                reward, info = evaluate_review(action, repo, issues, self.state.step_count)
                obs_result = f"Review submitted. Feedback: {info.get('feedback', 'Processed.')}"
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
            # Absolute guardrail: If anything fails, return a safe JSON instead of crashing FastAPI
            obs = CodeObservation(
                task_id=self.state.task_id if self.current_task_data else "error",
                context="Error state", available_files=[], action_result=f"Internal Env Error: {str(e)}",
                step_number=self.state.step_count, done=False, reward=0.0
            )
            return obs, 0.0, False, {"error": str(e)}
