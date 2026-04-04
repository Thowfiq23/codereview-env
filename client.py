import os
import requests
from typing import Tuple, Dict, Any
from models import AgentAction, CodeObservation

class CodeReviewEnv:
    """
    Python SDK Client for the CodeReview V2 Environment.
    """
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self) -> CodeObservation:
        """Resets the environment and returns the initial PR context."""
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return CodeObservation(**response.json())

    def step(self, action: AgentAction) -> Tuple[CodeObservation, float, bool, Dict[str, Any]]:
        """
        Takes a step in the environment using the provided AgentAction.
        Returns: (observation, reward, done, info)
        """
        # Convert the Pydantic model to a JSON string
        action_json = action.model_dump_json()
        
        response = requests.post(
            f"{self.base_url}/step", 
            data=action_json.encode('utf-8')
        )
        response.raise_for_status()
        
        data = response.json()
        obs = CodeObservation(**data["observation"])
        reward = float(data["reward"])
        done = bool(data["done"])
        info = dict(data["info"])
        
        return obs, reward, done, info
