"""
codereview_env/server/environment.py
"""
from uuid import uuid4
from ..models import CodeObservation, ReviewAction, ReviewState, StepResponse
from ..tasks import TASKS
from ..grader import evaluate_review, parse_agent_action

class CodeReviewEnvironment:
    def __init__(self): self._state = ReviewState()
    
    def _obs(self, r=0.0, f=None):
        if self._state.done or self._state.current_task_index >= len(TASKS):
            return CodeObservation(task_id="done", task_description="", code_snippet="", filename="", done=True, reward=r, feedback=f)
        t = TASKS[self._state.current_task_index]
        return CodeObservation(task_id=t["id"], task_description=t["task_description"], code_snippet=t["code_snippet"], filename=t["filename"], done=False, reward=r, feedback=f)
        
    def reset(self):
        self._state = ReviewState(episode_id=str(uuid4()), current_task_index=0, done=False)
        return self._obs()
        
    def step(self, raw_action: str):
        if self._state.done: return StepResponse(observation=self._obs(), reward=0.0, done=True)
        act = parse_agent_action(raw_action)
        if not act: return StepResponse(observation=self._obs(0.0, "Invalid JSON"), reward=0.0, done=self._state.done, info={"error": "Parse fail"})
        
        t = TASKS[self._state.current_task_index]
        rew, info = evaluate_review(act, t["code_snippet"], t["ground_truth_issues"])
        self._state.current_task_index += 1
        if self._state.current_task_index >= len(TASKS): self._state.done = True
        return StepResponse(observation=self._obs(rew, f"Graded TP:{info.get(true_positives)}"), reward=rew, done=self._state.done, info=info)
        
    def state(self): return self._state.model_dump()
