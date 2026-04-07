"""
FastAPI server exposing the Village Microgrid environment via HTTP.
Deploy on Hugging Face Spaces (SDK: docker, port 7860).

Endpoints
---------
GET  /             — health check / landing page
POST /reset        — start a new episode, returns initial Observation
POST /step         — advance one step, returns (obs, reward, done, info)
GET  /state        — return current internal state snapshot
GET  /tasks        — list supported task names
POST /grade        — grade the current completed episode
"""

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import Action, VillageMicrogridEnv

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Village Microgrid Energy Management",
    description=(
        "OpenEnv-compatible RL environment for microgrid control. "
        "One step = one hour. Tasks: summer day, winter night, rolling blackout."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateful per-session server)
env = VillageMicrogridEnv()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "task1_summer_day"


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class GradeResponse(BaseModel):
    score: float
    task_name: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Health-check endpoint.  Must return 200 for automated validation."""
    return {"status": "ok", "environment": "VillageMicrogridEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """Return all supported task names."""
    return {"tasks": VillageMicrogridEnv.SUPPORTED_TASKS}


@app.post("/reset")
def reset_env(request: ResetRequest):
    """
    Start a new episode.

    Body
    ----
    ``task_name`` : one of ``task1_summer_day`` | ``task2_winter_night``
                   | ``task3_rolling_blackout``

    Returns the initial Observation for step 0.
    """
    try:
        obs = env.reset(task_name=request.task_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs.model_dump()


@app.post("/step", response_model=StepResponse)
def step_env(request: StepRequest):
    """
    Advance the simulation by one hour step.

    Returns the next Observation, step reward, done flag, and info dict.
    """
    if not env._task_name:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if env._current_step >= env._max_steps:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset to start again.")
    obs, reward, done, info = env.step(request.action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def get_state():
    """Return the full internal state of the environment."""
    if not env._task_name:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return env.state()


@app.post("/grade", response_model=GradeResponse)
def grade_episode():
    """Grade the current completed episode.  Must be called after done=True."""
    if not env._task_name:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    score = env.grade_episode()
    return GradeResponse(score=score, task_name=env._task_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
