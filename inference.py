"""
inference.py — Baseline inference script for Village Microgrid Energy Management
==================================================================================
Uses an LLM (via OpenAI-compatible API) to decide each step's action.

The LLM receives the current observation as a JSON prompt and returns a JSON
action. A fallback heuristic is used if the model output cannot be parsed.

Log format (MANDATORY — do not modify):
  [START] {"task": ..., "timestamp": ...}
  [STEP]  {"step": ..., "observation": ..., "action": ..., "reward": ..., "done": ...}
  [END]   {"task": ..., "score": ..., "total_reward": ..., "critical_failures": ...,
            "timestamp": ...}

Environment variables required
-------------------------------
  API_BASE_URL   The base URL of the OpenAI-compatible API endpoint.
  MODEL_NAME     The model identifier (e.g. "gpt-4o-mini").
  HF_TOKEN       Your Hugging Face / API key (used as bearer token).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from openai import OpenAI

from environment import Action, VillageMicrogridEnv

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print(
        "WARNING: HF_TOKEN is not set. API calls may fail.",
        file=sys.stderr,
    )

# OpenAI-compatible client (works with HF Inference, OpenAI, Together, etc.)
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ---------------------------------------------------------------------------
# LLM-based action selection
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a microgrid energy management controller for a small village.
At each hourly step you MUST return a JSON object (no markdown, no explanation)
with EXACTLY these keys:
  {
    "grid_import_kwh": <float, >= 0>,
    "battery_action_kwh": <float, positive=charge, negative=discharge>,
    "residential_supplied_kwh": <float, >= 0>
  }

Rules:
- Always cover critical_demand_kwh first.
- Discharge the battery (negative battery_action_kwh) when solar is low.
- Charge the battery when solar is abundant.
- Minimise grid_import_kwh to stay self-sufficient.
- residential_supplied_kwh must not exceed residential_demand_kwh.
"""


def get_llm_action(obs: Dict[str, Any], task_name: str) -> Action:
    """
    Ask the LLM for an action given the current observation.
    Falls back to a deterministic greedy heuristic on any parse error.
    """
    user_content = (
        f"Task: {task_name}\n"
        f"Observation: {json.dumps(obs, indent=2)}\n"
        "Return a JSON action object."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return Action(
            grid_import_kwh=float(data.get("grid_import_kwh", 0.0)),
            battery_action_kwh=float(data.get("battery_action_kwh", 0.0)),
            residential_supplied_kwh=float(data.get("residential_supplied_kwh", 0.0)),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] LLM parse error: {exc}. Using heuristic.", file=sys.stderr)
        return _heuristic_action(obs)


def _heuristic_action(obs: Dict[str, Any]) -> Action:
    """
    Simple greedy heuristic: discharge battery to cover demand, import only remainder.
    Used as fallback when the LLM returns an unparseable response.
    """
    battery_soc = obs["battery_soc_kwh"]
    solar = obs["solar_gen_kwh"]
    critical = obs["critical_demand_kwh"]
    residential = obs["residential_demand_kwh"]
    needed = critical + residential
    # Discharge up to 10 kWh (hardware rate limit)
    discharge = min(battery_soc, 10.0, max(0.0, needed - solar))
    available = solar + discharge
    grid_need = max(0.0, needed - available)
    return Action(
        grid_import_kwh=round(grid_need, 2),
        battery_action_kwh=round(-discharge, 2),
        residential_supplied_kwh=round(residential, 2),
    )


# ---------------------------------------------------------------------------
# Helper: structured stdout logging
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_start(task_name: str) -> None:
    print(
        "[START] "
        + json.dumps({"task": task_name, "timestamp": _ts()}),
        flush=True,
    )


def log_step(
    step: int,
    observation: Dict[str, Any],
    action: Dict[str, Any],
    reward: float,
    done: bool,
) -> None:
    print(
        "[STEP] "
        + json.dumps(
            {
                "step": step,
                "observation": observation,
                "action": action,
                "reward": round(reward, 4),
                "done": done,
            }
        ),
        flush=True,
    )


def log_end(
    task_name: str,
    score: float,
    total_reward: float,
    critical_failures: int,
) -> None:
    print(
        "[END] "
        + json.dumps(
            {
                "task": task_name,
                "score": round(score, 4),
                "total_reward": round(total_reward, 4),
                "critical_failures": critical_failures,
                "timestamp": _ts(),
            }
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(task_name: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Run a complete episode for ``task_name`` and return summary statistics.

    Parameters
    ----------
    task_name : str
        One of the supported VillageMicrogridEnv task names.
    use_llm : bool
        If True, query the LLM for each step. If False, use heuristic only.
    """
    env = VillageMicrogridEnv()
    obs_obj = env.reset(task_name)
    obs = obs_obj.model_dump()

    log_start(task_name)

    done = False
    step = 0

    while not done:
        if use_llm:
            action = get_llm_action(obs, task_name)
        else:
            action = _heuristic_action(obs)

        obs_obj, reward, done, info = env.step(action)
        next_obs = obs_obj.model_dump()

        log_step(
            step=step,
            observation=obs,
            action=action.model_dump(),
            reward=reward,
            done=done,
        )

        obs = next_obs
        step += 1

    score = env.grade_episode()
    state = env.state()
    total_reward = state["total_reward"]
    critical_failures = state["critical_failures"]

    log_end(task_name, score, total_reward, critical_failures)

    return {
        "task": task_name,
        "score": score,
        "total_reward": total_reward,
        "critical_failures": critical_failures,
        "steps": step,
    }


# ---------------------------------------------------------------------------
# Entry point — run all 3 tasks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Set use_llm=False to run in pure-heuristic mode (no API needed).
    # Set use_llm=True to use the LLM defined by API_BASE_URL / MODEL_NAME.
    use_llm = bool(HF_TOKEN)

    results = []
    for task in VillageMicrogridEnv.SUPPORTED_TASKS:
        summary = run_task(task_name=task, use_llm=use_llm)
        results.append(summary)

    # Final summary table
    print("\n" + "=" * 60, flush=True)
    print("FINAL SCORES", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(
            f"  {r['task']:35s}  score={r['score']:.4f}  "
            f"reward={r['total_reward']:8.2f}  blackouts={r['critical_failures']}",
            flush=True,
        )
    print("=" * 60, flush=True)
