"""
codereview_env/server/app.py
"""
from fastapi import FastAPI, Request
from ..models import CodeObservation, StepResponse
from .environment import CodeReviewEnvironment

app = FastAPI()
env = CodeReviewEnvironment()

@app.get("/health")
async def health(): return {"status": "ok"}

@app.post("/reset", response_model=CodeObservation)
async def reset(): return env.reset()

@app.post("/step", response_model=StepResponse)
async def step(req: Request):
    try: raw = (await req.body()).decode("utf-8")
    except: raw = ""
    return env.step(raw)

@app.get("/state")
async def state(): return env.state()
