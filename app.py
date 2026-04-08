"""
FastAPI server exposing the Disaster Relief Coordination environment via HTTP.
Deploy on Hugging Face Spaces (SDK: docker, port 7860).

Endpoints
---------
GET  /             — health check / landing page
GET  /tasks        — list supported task names
POST /reset        — start a new episode, returns initial Observation
POST /step         — advance one step, returns StepResult (obs, reward, done, info)
GET  /state        — return current internal state snapshot
POST /grade        — grade the current completed episode
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.env.environment import DisasterReliefEnv

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Disaster Relief Coordination",
    description=(
        "OpenEnv-compatible environment for disaster relief coordination. "
        "An LLM coordinator triages reports, allocates limited resources, "
        "and resolves incidents under uncertainty."
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
env = DisasterReliefEnv()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "task1_flood_easy"
    seed: int = 42


class StepRequest(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


class GradeResponse(BaseModel):
    score: float
    task_name: str
    total_reward: float
    reports_resolved: int
    reports_expired: int
    critical_missed: int
    reports_false_flagged: int
    total_reports: int
    steps_used: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Health-check endpoint. Must return 200 for automated validation."""
    return {
        "status": "ok",
        "environment": "DisasterReliefCoordination",
        "version": "1.0.0",
    }


@app.get("/tasks")
def list_tasks():
    """Return all supported task names."""
    return {"tasks": DisasterReliefEnv.supported_tasks()}


@app.post("/reset")
def reset_env(request: ResetRequest):
    """
    Start a new episode.

    Body
    ----
    ``task_name`` : one of ``task1_flood_easy`` | ``task2_storm_medium``
                   | ``task3_cascade_hard``
    ``seed`` : int (default 42)

    Returns the initial Observation for step 0.
    """
    try:
        obs = env.reset(task_name=request.task_name, seed=request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step")
def step_env(request: StepRequest):
    """
    Advance the simulation by one step.

    Body
    ----
    ``tool`` : str — name of the tool to invoke
    ``args`` : dict — arguments for the tool

    Returns StepResult with observation, reward, done, info.
    """
    if not env.is_initialized:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if env.is_done:
        raise HTTPException(
            status_code=400,
            detail="Episode done. Call /reset to start again.",
        )
    result = env.step({"tool": request.tool, "args": request.args})
    return result


@app.get("/state")
def get_state():
    """Return the full internal state of the environment."""
    if not env.is_initialized:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return env.get_state()


@app.post("/grade")
def grade_episode():
    """Grade the current completed episode. Must be called after done=True."""
    if not env.is_initialized:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    result = env.grade()
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
