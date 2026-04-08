"""
inference.py — Baseline inference script for Disaster Relief Coordination
==========================================================================
Uses an LLM (via OpenAI-compatible API) to decide each step's action.

The LLM receives the current observation as a structured prompt and returns
a JSON tool call. A fallback heuristic is used if the model output cannot
be parsed.

Log format (MANDATORY — do not modify):
  [START] {"task": ..., "timestamp": ...}
  [STEP]  {"step": ..., "observation": ..., "action": ..., "reward": ..., "done": ...}
  [END]   {"task": ..., "score": ..., "total_reward": ..., "critical_missed": ...,
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
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.env.environment import DisasterReliefEnv

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
You are a disaster relief coordination AI. You manage incoming disaster reports,
allocate limited resources, and resolve incidents under time pressure.

At each step you MUST return a JSON object (no markdown, no explanation) with:
  {"tool": "<tool_name>", "args": {<tool_arguments>}}

Available tools:
- call_intake_agent: Process a report through classification, urgency, and verification.
  Args: {"report_id": "RPT-XXX"}
- send_resource: Dispatch a resource to a report.
  Args: {"resource_id": "RES-XXX", "report_id": "RPT-XXX"}
- call_dispatch_agent: Query inventory then dispatch.
  Args: {"resource_id": "RES-XXX", "report_id": "RPT-XXX"}
- mark_false_report: Flag a report as false alarm.
  Args: {"report_id": "RPT-XXX", "reason": "description"}
- check_operation: Check status of a report or assignment.
  Args: {"target_id": "RPT-XXX or ASG-XXX"}
- call_monitor_agent: Monitor and optionally close a case.
  Args: {"target_id": "RPT-XXX or ASG-XXX"}
- close_case: Close a completed assignment.
  Args: {"assignment_id": "ASG-XXX"}
- get_resources: List all resources and their status.
  Args: {}
- classify_report: Classify a single report's category.
  Args: {"report_id": "RPT-XXX"}
- assess_report_urgency: Score a report's urgency.
  Args: {"report_id": "RPT-XXX"}
- verify_report: Verify a report's authenticity.
  Args: {"report_id": "RPT-XXX"}
- reroute_resource: Reroute a deployed resource.
  Args: {"resource_id": "RES-XXX"}

Strategy priorities:
1. Process unclassified reports first (call_intake_agent)
2. Flag reports with low verification confidence (<0.3) as false
3. Dispatch flood-capable resources (boats, helicopters) to zones with flood_depth >= 2
4. Prioritize critical reports with approaching deadlines
5. Monitor active assignments periodically
6. Match resource types to incident categories when possible
"""


def get_llm_action(obs: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """
    Ask the LLM for an action given the current observation.
    Falls back to a deterministic heuristic on any parse error.
    """
    # Build a concise observation summary to fit context
    summary = _summarize_observation(obs)

    user_content = (
        f"Task: {task_name}\n"
        f"Step: {obs['step']}/{obs['max_steps']}\n\n"
        f"{summary}\n\n"
        "Return a JSON tool call."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        tool = data.get("tool", "")
        args = data.get("args", {})
        if not tool:
            raise ValueError("Missing 'tool' in LLM response")
        return {"tool": tool, "args": args}
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] LLM parse error: {exc}. Using heuristic.", file=sys.stderr)
        return _heuristic_action(obs)


def _summarize_observation(obs: Dict[str, Any]) -> str:
    """Build a concise text summary of the observation for the LLM."""
    lines = []

    # Zones
    zones = obs.get("zones", [])
    if zones:
        zone_info = {z["id"]: z for z in zones}
        lines.append("Zones:")
        for z in zones:
            flags = []
            if z["flood_depth_level"] >= 2:
                flags.append(f"FLOODED(depth={z['flood_depth_level']})")
            if z["access_blocked"]:
                flags.append("BLOCKED")
            if z["comms_blackout"]:
                flags.append("NO_COMMS")
            lines.append(f"  {z['id']} ({z['name']}): severity={z['severity']} {' '.join(flags)}")
    else:
        zone_info = {}

    # Pending reports
    pending = obs.get("pending_reports", [])
    if pending:
        lines.append(f"\nPending reports ({len(pending)}):")
        for r in pending[:8]:  # limit to 8 to fit context
            crit = "CRITICAL " if r["is_critical"] else ""
            dl = f" deadline=step{r['deadline_step']}" if r.get("deadline_step") else ""
            cat = r["category"] if r["classified"] else "unclassified"
            ver = ""
            if r["verified"] and r.get("verification_confidence") is not None:
                ver = f" conf={r['verification_confidence']:.1f}"
            assigned = f" assigned={r['assigned_resource_id']}" if r.get("assigned_resource_id") else ""
            lines.append(
                f"  {r['id']}: {crit}{cat} zone={r['zone_id']} "
                f"urgency={r['urgency']} status={r['status']}{dl}{ver}{assigned}"
            )
        if len(pending) > 8:
            lines.append(f"  ... and {len(pending) - 8} more")

    # Active assignments
    assignments = obs.get("active_assignments", [])
    if assignments:
        lines.append(f"\nActive assignments ({len(assignments)}):")
        for a in assignments[:5]:
            stuck = " STUCK" if a["stuck"] else ""
            lines.append(
                f"  {a['id']}: {a['resource_id']}→{a['report_id']} "
                f"status={a['status']} ETA=step{a['expected_completion_step']}{stuck}"
            )

    # Resources
    resources = obs.get("available_resources", [])
    avail = [r for r in resources if r["status"] == "available"]
    deployed = [r for r in resources if r["status"] == "deployed"]
    lines.append(f"\nResources: {len(avail)} available, {len(deployed)} deployed, {len(resources)} total")
    if avail:
        for r in avail:
            flood = " flood-capable" if r.get("can_traverse_flood") else ""
            lines.append(f"  {r['id']}: {r['type']} [available]{flood}")

    # Warnings
    warnings = obs.get("warnings", [])
    if warnings:
        lines.append("\nWARNINGS:")
        for w in warnings:
            lines.append(f"  ⚠ {w}")

    # Last action feedback
    if obs.get("last_action_result"):
        lines.append(f"\nLast result: {obs['last_action_result'][:150]}")
    if obs.get("last_action_error"):
        lines.append(f"\nLast error: {obs['last_action_error'][:150]}")

    return "\n".join(lines)


def _heuristic_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic flood-aware heuristic. Used as fallback when the LLM
    returns an unparseable response, or for offline testing.
    """
    pending = obs.get("pending_reports", [])
    resources = obs.get("available_resources", [])
    assignments = obs.get("active_assignments", [])
    zones = {z["id"]: z for z in obs.get("zones", [])}

    if not pending:
        # Nothing to do — monitor an active assignment if any
        if assignments:
            return {"tool": "check_operation", "args": {"target_id": assignments[0]["report_id"]}}
        return {"tool": "get_resources", "args": {}}

    # Pick highest priority pending report
    rpt = pending[0]

    # Step 1: Intake unclassified reports
    if not rpt["classified"]:
        return {"tool": "call_intake_agent", "args": {"report_id": rpt["id"]}}

    # Step 2: Flag low-confidence reports as false
    if rpt["verified"] and rpt.get("verification_confidence", 1.0) < 0.3:
        return {"tool": "mark_false_report", "args": {"report_id": rpt["id"], "reason": "low verification confidence"}}

    # Step 3: Dispatch — flood-aware and type-aware
    if rpt.get("assigned_resource_id") is None and rpt["status"] in ("pending", "triaged"):
        zone = zones.get(rpt.get("zone_id", ""), {})
        flood_depth = zone.get("flood_depth_level", 0)

        avail = [r for r in resources if r["status"] == "available"]

        # Filter for flood capability if needed
        if flood_depth >= 2:
            flood_avail = [r for r in avail if r.get("can_traverse_flood", False)]
            if flood_avail:
                avail = flood_avail
            else:
                # No flood-capable resource — skip to next report or monitor
                if len(pending) > 1:
                    next_rpt = pending[1]
                    if not next_rpt["classified"]:
                        return {"tool": "call_intake_agent", "args": {"report_id": next_rpt["id"]}}
                return {"tool": "check_operation", "args": {"target_id": rpt["id"]}}

        if avail:
            return {"tool": "send_resource", "args": {"resource_id": avail[0]["id"], "report_id": rpt["id"]}}

    # Step 4: Monitor if nothing else
    if assignments:
        return {"tool": "check_operation", "args": {"target_id": assignments[0]["report_id"]}}

    return {"tool": "get_resources", "args": {}}


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
    critical_missed: int,
) -> None:
    print(
        "[END] "
        + json.dumps(
            {
                "task": task_name,
                "score": round(score, 4),
                "total_reward": round(total_reward, 4),
                "critical_missed": critical_missed,
                "timestamp": _ts(),
            }
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(task_name: str, seed: int = 42, use_llm: bool = True) -> Dict[str, Any]:
    """
    Run a complete episode for ``task_name`` and return summary statistics.

    Parameters
    ----------
    task_name : str
        One of the supported DisasterReliefEnv task names.
    seed : int
        Random seed for deterministic scenario generation.
    use_llm : bool
        If True, query the LLM for each step. If False, use heuristic only.
    """
    env = DisasterReliefEnv()
    obs = env.reset(task_name, seed=seed)

    log_start(task_name)

    done = False
    step = 0

    while not done:
        if use_llm:
            action = get_llm_action(obs, task_name)
        else:
            action = _heuristic_action(obs)

        result = env.step(action)
        next_obs = result["observation"]
        reward = result["reward"]
        done = result["done"]

        log_step(
            step=step,
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )

        obs = next_obs
        step += 1

    grade = env.grade()
    score = grade["score"]
    total_reward = grade["total_reward"]
    critical_missed = grade["critical_missed"]

    log_end(task_name, score, total_reward, critical_missed)

    return {
        "task": task_name,
        "score": score,
        "total_reward": total_reward,
        "critical_missed": critical_missed,
        "reports_resolved": grade["reports_resolved"],
        "total_reports": grade["total_reports"],
        "steps": step,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all tasks and print summary."""
    tasks = DisasterReliefEnv.supported_tasks()

    # Check for --heuristic-only flag
    use_llm = "--heuristic-only" not in sys.argv

    if not use_llm:
        print("Running in heuristic-only mode (no LLM).", file=sys.stderr)

    results = []
    for task in tasks:
        result = run_task(task, use_llm=use_llm)
        results.append(result)

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(
            f"  {r['task']}: score={r['score']:.4f}  "
            f"reward={r['total_reward']:.1f}  "
            f"resolved={r['reports_resolved']}/{r['total_reports']}  "
            f"critical_missed={r['critical_missed']}  "
            f"steps={r['steps']}",
            flush=True,
        )


if __name__ == "__main__":
    main()


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
