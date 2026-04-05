import requests
from typing import Tuple, Dict, Any
from models import CodeObservation, AgentAction, StepResponse

class CodeReviewEnv:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url

    def reset(self) -> CodeObservation:
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return CodeObservation.model_validate(response.json())

    def step(self, action: AgentAction | str) -> Tuple[CodeObservation, float, bool, Dict[str, Any]]:
        # Handle both Pydantic objects and raw JSON strings
        if isinstance(action, str):
            payload = action
        else:
            payload = action.model_dump_json()
            
        response = requests.post(
            f"{self.base_url}/step",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        data = response.json()
        obs = CodeObservation.model_validate(data["observation"])
        return obs, data["reward"], data["done"], data["info"]

    def state(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()