"""
codereview_env/server/environment.py

Environment state management. Handles OpenEnv step, reset, and state tracking.
Enforces zero-crash policy on malformed actions.
"""
from typing import Any, Dict
from uuid import uuid4

from models import CodeObservation, ReviewAction, ReviewState, StepResponse
from tasks import TASKS
from grader import evaluate_review, parse_agent_action

class CodeReviewEnvironment:
    """
    Maintains episode state and progression through the dataset.
    Follows strictly OpenEnv spec mechanics.
    """
    def __init__(self):
        self._state = ReviewState()

    def _get_current_observation(self, reward: float = 0.0, feedback: str | None = None) -> CodeObservation:
        task_idx = self._state.current_task_index
        
        # If episode is done, return a terminal observation
        if self._state.done or task_idx >= len(TASKS):
            return CodeObservation(
                task_id="done",
                task_description="Episode complete.",
                code_snippet="",
                filename="",
                language="",
                context="",
                step_number=self._state.step_count,
                feedback=feedback,
                done=True,
                reward=reward
            )

        task = TASKS[task_idx]
        return CodeObservation(
            task_id=task["id"],
            task_description=task["task_description"],
            code_snippet=task["code_snippet"],
            filename=task["filename"],
            language=task.get("language", "python"),
            context=task.get("context", ""),
            step_number=self._state.step_count,
            feedback=feedback,
            done=False,
            reward=reward
        )

    def reset(self) -> CodeObservation:
        """Starts a new episode and returns the first observation."""
        self._state = ReviewState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task_index=0,
            total_reward=0.0,
            done=False
        )
        if TASKS:
            self._state.task_id = TASKS[0]["id"]
            
        return self._get_current_observation(reward=0.0, feedback=None)

    def step(self, raw_action: str) -> StepResponse:
        """
        Consumes an agent action, evaluates it, and advances the state.
        Zero-Crash: Malformed JSON or schema errors result in 0.0 reward and retry feedback.
        """
        if self._state.done:
            obs = self._get_current_observation(reward=0.0, feedback="Episode already completed.")
            return StepResponse(observation=obs, reward=0.0, done=True, info={"warn": "Episode already done"})

        # 1. Unbreakable parsing
        action: ReviewAction | None = parse_agent_action(raw_action)
        if action is None:
            # We do NOT advance the task index. The agent gets 0 reward and must try again.
            self._state.step_count += 1
            error_feedback = "Invalid JSON/Action format. Please submit valid JSON conforming to the ReviewAction schema."
            obs = self._get_current_observation(reward=0.0, feedback=error_feedback)
            return StepResponse(
                observation=obs,
                reward=0.0,
                done=self._state.done,
                info={"error": error_feedback}
            )

        # 2. Extract current task data for evaluating
        current_task = TASKS[self._state.current_task_index]

        # 3. Evaluate the review action deterministically
        reward, grader_info = evaluate_review(
            agent_action=action,
            code_snippet=current_task["code_snippet"],
            ground_truth_issues=current_task["ground_truth_issues"]
        )

        # Build feedback string from grader info
        tp = grader_info.get("true_positives", 0)
        fp = grader_info.get("false_positives", 0)
        total_issues = len(current_task.get("ground_truth_issues", []))
        feedback = f"Graded. Found {tp}/{total_issues} issues. False positives: {fp}. Penalty multiplier: {grader_info.get('penalty_multiplier', 1.0):.2f}"

        # 4. Advance State
        self._state.step_count += 1
        self._state.current_task_index += 1
        self._state.total_reward += reward

        if self._state.current_task_index >= len(TASKS):
            self._state.done = True
            self._state.task_id = ""
        else:
            self._state.task_id = TASKS[self._state.current_task_index]["id"]

        # 5. Get NEXT observation
        next_obs = self._get_current_observation(reward=reward, feedback=feedback)

        # Add total reward to info for monitoring
        grader_info["episode_total_reward"] = self._state.total_reward

        return StepResponse(
            observation=next_obs,
            reward=reward,
            done=self._state.done,
            info=grader_info
        )

    def state(self) -> Dict[str, Any]:
        """Returns the current episode state dictionary."""
        return self._state.model_dump()
