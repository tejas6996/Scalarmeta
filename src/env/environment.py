"""
Disaster Relief Coordination — Main Environment Class
=======================================================
Wires together scenario generation, world state, tool execution,
reward computation, observation building, and grading into a single
``DisasterReliefEnv`` with the standard OpenEnv interface:

    reset(task_name) → Observation
    step(action)     → StepResult
    state()          → EnvironmentState
    grade_episode()  → float
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.env.graders import grade_episode as _grade
from src.env.models import (
    Action,
    EnvironmentState,
    Observation,
    Reward,
    StepResult,
)
from src.env.observation import build_observation
from src.env.rewards import compute_no_action_reward, compute_step_reward
from src.env.scenarios import SUPPORTED_TASKS, generate_scenario
from src.env.state import WorldState
from src.env.tool_registry import execute_tool, get_tool_names


class DisasterReliefEnv:
    """OpenEnv-compatible disaster relief coordination environment."""

    def __init__(self) -> None:
        self._state: Optional[WorldState] = None
        self._last_result: Optional[str] = None
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_name: str, seed: int = 42) -> Dict[str, Any]:
        """
        Start a new episode for the given task.

        Parameters
        ----------
        task_name : str
            One of the supported task names.
        seed : int
            Random seed for deterministic scenario generation.

        Returns
        -------
        dict
            JSON-serializable initial observation.
        """
        if task_name not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from {SUPPORTED_TASKS}."
            )

        scenario = generate_scenario(task_name, seed=seed)
        self._state = WorldState.from_scenario(scenario)
        self._last_result = None
        self._last_error = None

        # Run step 0 advance to surface initial reports
        self._state.current_step = 0
        self._state.advance_time()

        obs = build_observation(self._state)
        return obs.model_dump()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step: parse action → run tool → compute reward →
        advance time → build next observation.

        Parameters
        ----------
        action : dict
            Must have ``"tool"`` (str) and optionally ``"args"`` (dict).

        Returns
        -------
        dict
            JSON-serializable StepResult with observation, reward, done, info.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new one.")

        state = self._state
        self._last_result = None
        self._last_error = None

        # --- Parse action ---
        tool_name = action.get("tool", "")
        tool_args = action.get("args", {})

        if not tool_name or not isinstance(tool_name, str):
            # Malformed action
            reward = compute_no_action_reward(state)
            self._last_error = "Malformed action: missing or invalid 'tool' field."
        else:
            # Execute tool
            result_text, reward_delta = execute_tool(state, tool_name, tool_args)
            reward = compute_step_reward(
                state, tool_name, tool_args, result_text, reward_delta
            )

            if result_text.startswith("Error:"):
                self._last_error = result_text
                self._last_result = None
            else:
                self._last_result = result_text
                self._last_error = None

        # --- Advance time ---
        state.current_step += 1
        if state.current_step < state.max_steps:
            state.advance_time()
        else:
            state.done = True

        # --- Build observation ---
        obs = build_observation(
            state,
            last_action_result=self._last_result,
            last_action_error=self._last_error,
        )

        step_result = StepResult(
            observation=obs,
            reward=reward.total,
            done=state.done,
            info={
                "step": state.current_step,
                "total_reward": state.total_reward,
                "reward_breakdown": reward.model_dump(),
            },
        )
        return step_result.model_dump()

    def get_state(self) -> Dict[str, Any]:
        """
        Return the public environment state snapshot (no ground truth).

        Returns
        -------
        dict
            JSON-serializable EnvironmentState.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        state = self._state
        env_state = EnvironmentState(
            current_step=state.current_step,
            max_steps=state.max_steps,
            task_name=state.task_name,
            total_reward=state.total_reward,
            total_reports=state.total_reports,
            reports_resolved=state.reports_resolved,
            reports_expired=state.reports_expired,
            reports_false_flagged=state.reports_false_flagged,
            critical_reports_total=state.critical_total,
            critical_reports_missed=state.critical_missed,
            active_assignments=state.active_assignments_count,
            resources_available=state.resources_available_count,
            resources_deployed=state.resources_deployed_count,
        )
        return env_state.model_dump()

    def grade(self) -> Dict[str, Any]:
        """
        Grade the completed episode.

        Returns
        -------
        dict
            ``{"score": float, "task_name": str, ...}``
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        score = _grade(self._state)
        return {
            "score": score,
            "task_name": self._state.task_name,
            "total_reward": self._state.total_reward,
            "reports_resolved": self._state.reports_resolved,
            "reports_expired": self._state.reports_expired,
            "critical_missed": self._state.critical_missed,
            "reports_false_flagged": self._state.reports_false_flagged,
            "total_reports": self._state.total_reports,
            "steps_used": self._state.current_step,
        }

    def close(self) -> None:
        """Clean up resources."""
        self._state = None
        self._last_result = None
        self._last_error = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    @property
    def is_done(self) -> bool:
        return self._state is not None and self._state.done

    @staticmethod
    def supported_tasks():
        return list(SUPPORTED_TASKS)
