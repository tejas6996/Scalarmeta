"""
Disaster Relief Coordination — Reward Function
================================================
Computes per-step reward, handles repeat detection, deadline-miss penalties,
and logs actions to episode memory.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from src.env.models import EpisodeMemoryEntry, Reward
from src.env.state import WorldState


# ---------------------------------------------------------------------------
# Tool category mapping (for structured reward buckets)
# ---------------------------------------------------------------------------

_TRIAGE_TOOLS = {"classify_report", "assess_report_urgency", "verify_report", "call_intake_agent"}
_DISPATCH_TOOLS = {"get_resources", "send_resource", "reroute_resource", "call_dispatch_agent"}
_MONITOR_TOOLS = {"check_operation", "close_case", "mark_false_report", "call_monitor_agent"}

BASE_REWARD = 0.5
REPEAT_PENALTY = -0.5


def compute_step_reward(
    state: WorldState,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_result: str,
    tool_reward_delta: float,
) -> Reward:
    """
    Compute the full structured reward for a single step.

    Parameters
    ----------
    state : WorldState
        Current world state (already mutated by the tool).
    tool_name : str
        Name of the tool that was executed.
    tool_args : dict
        Arguments passed to the tool.
    tool_result : str
        Result text returned by the tool.
    tool_reward_delta : float
        Raw reward delta returned by the tool handler.

    Returns
    -------
    Reward
        Structured reward breakdown.
    """
    reward = Reward(base=BASE_REWARD)

    # --- Categorize into reward buckets ---
    if tool_name in _TRIAGE_TOOLS:
        reward.triage_reward = tool_reward_delta
    elif tool_name in _DISPATCH_TOOLS:
        reward.dispatch_reward = tool_reward_delta
    elif tool_name in _MONITOR_TOOLS:
        reward.monitor_reward = tool_reward_delta
    else:
        # Unknown tool — should have been caught by registry, but handle gracefully
        reward.penalty = tool_reward_delta if tool_reward_delta < 0 else 0.0

    # --- Repeat detection ---
    action_key = _make_action_key(tool_name, tool_args)
    if action_key == state.last_action_key:
        reward.penalty += REPEAT_PENALTY
    state.last_action_key = action_key

    # --- Deadline-miss penalties (from advance_time's recent_changes) ---
    deadline_misses = sum(
        1 for c in state.recent_changes if "EXPIRED" in c
    )
    critical_misses = sum(
        1 for c in state.recent_changes
        if "EXPIRED" in c and any(
            state.reports[rid].is_critical
            for rid in state.reports
            if rid in c and state.reports[rid].is_critical
        )
    )
    # -2.0 per critical miss, -0.5 per non-critical miss
    if critical_misses > 0:
        reward.penalty += -2.0 * critical_misses
    non_critical_misses = deadline_misses - critical_misses
    if non_critical_misses > 0:
        reward.penalty += -0.5 * non_critical_misses

    # --- Compute total ---
    reward.total = (
        reward.base
        + reward.triage_reward
        + reward.dispatch_reward
        + reward.monitor_reward
        + reward.penalty
    )

    # --- Update state bookkeeping ---
    state.total_reward += reward.total
    state.step_rewards.append(reward.total)

    # --- Log to episode memory ---
    _log_memory(state, tool_name, tool_args, tool_result, reward)

    return reward


def compute_no_action_reward(state: WorldState) -> Reward:
    """
    Compute reward for a step where no valid action was taken
    (malformed, timeout, etc.).
    """
    reward = Reward(
        base=BASE_REWARD,
        penalty=-0.5,
        total=BASE_REWARD - 0.5,
    )
    state.total_reward += reward.total
    state.step_rewards.append(reward.total)
    return reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_action_key(tool_name: str, args: Dict[str, Any]) -> str:
    """Create a deterministic string key for repeat detection."""
    sorted_args = json.dumps(args, sort_keys=True, default=str)
    return f"{tool_name}:{sorted_args}"


def _log_memory(
    state: WorldState,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_result: str,
    reward: Reward,
) -> None:
    """Append an EpisodeMemoryEntry for this step."""
    # Determine actor from tool category
    if tool_name in _TRIAGE_TOOLS:
        actor = "intake"
    elif tool_name in _DISPATCH_TOOLS:
        actor = "dispatch"
    elif tool_name in _MONITOR_TOOLS:
        actor = "monitor"
    else:
        actor = "coordinator"

    # Determine result status
    if tool_result.startswith("Error:"):
        result_status = "error"
    elif reward.total < 0:
        result_status = "partial"
    else:
        result_status = "success"

    # Extract important entity IDs from args
    entities = []
    for key in ("report_id", "resource_id", "target_id"):
        if key in tool_args:
            entities.append(tool_args[key])

    # Truncate result for memory (keep it compact)
    summary = tool_result[:200] if len(tool_result) > 200 else tool_result

    entry = EpisodeMemoryEntry(
        step=state.current_step,
        actor=actor,
        tool_name=tool_name,
        input_args=tool_args,
        summary=summary,
        result_status=result_status,
        reward=reward.total,
        important_entities=entities,
    )
    state.add_memory(entry)
