"""
Disaster Relief Coordination — Coordinator (Delegation) Tools
==============================================================
High-level tools that chain intake/dispatch/monitor tools internally.
These simulate sub-agent delegation — the coordinator tells a specialist
to handle a task, and gets back a brief summary.
Each returns (result_text, reward_delta).
"""

from __future__ import annotations

from typing import Tuple

from src.env.state import WorldState
from src.env.tools_intake import assess_report_urgency, classify_report, verify_report
from src.env.tools_dispatch import get_resources, send_resource
from src.env.tools_monitor import check_operation, close_case


def call_intake_agent(
    state: WorldState,
    report_id: str,
    instruction: str = "",
) -> Tuple[str, float]:
    """
    Delegate full intake processing of a report to the intake sub-agent.

    Chains: classify_report → assess_report_urgency → verify_report.
    Returns a consolidated brief and the total reward.
    """
    total_reward = 0.0
    briefs = []

    # Step 1: Classify
    result, reward = classify_report(state, report_id)
    total_reward += reward
    briefs.append(f"[Classify] {result}")

    # Check for errors
    if result.startswith("Error:"):
        return result, total_reward

    # Step 2: Assess urgency
    result, reward = assess_report_urgency(state, report_id)
    total_reward += reward
    briefs.append(f"[Urgency] {result}")

    # Step 3: Verify
    result, reward = verify_report(state, report_id)
    total_reward += reward
    briefs.append(f"[Verify] {result}")

    brief = f"Intake Agent Report for {report_id}:\n" + "\n".join(briefs)
    if instruction:
        brief += f"\n[Coordinator note: {instruction}]"

    return brief, total_reward


def call_dispatch_agent(
    state: WorldState,
    resource_id: str,
    report_id: str,
) -> Tuple[str, float]:
    """
    Delegate dispatch to the dispatch sub-agent.

    Chains: get_resources (info) → send_resource.
    Returns a brief and the reward from the dispatch.
    """
    # Step 1: Get current inventory (info only, no reward)
    inventory, _ = get_resources(state)

    # Step 2: Send the resource
    result, reward = send_resource(state, resource_id, report_id)

    brief = f"Dispatch Agent Brief:\n{inventory}\n\n[Action] {result}"
    return brief, reward


def call_monitor_agent(
    state: WorldState,
    target_id: str,
    instruction: str = "",
) -> Tuple[str, float]:
    """
    Delegate monitoring to the monitor sub-agent.

    Chains: check_operation → (close_case if instruction says "close").
    Returns a brief and the total reward.
    """
    total_reward = 0.0
    briefs = []

    # Step 1: Check current status
    result, reward = check_operation(state, target_id)
    total_reward += reward
    briefs.append(f"[Status] {result}")

    # Step 2: Auto-close if instructed
    instruction_lower = instruction.lower() if instruction else ""
    if "close" in instruction_lower or "resolve" in instruction_lower:
        close_result, close_reward = close_case(state, target_id, instruction)
        total_reward += close_reward
        briefs.append(f"[Close] {close_result}")

    brief = f"Monitor Agent Brief for {target_id}:\n" + "\n".join(briefs)
    if instruction and "close" not in instruction_lower and "resolve" not in instruction_lower:
        brief += f"\n[Coordinator note: {instruction}]"

    return brief, total_reward
