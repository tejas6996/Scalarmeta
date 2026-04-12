"""
inference.py — Baseline inference script for Disaster Relief Coordination
==========================================================================
Uses an LLM (via OpenAI-compatible API) to decide each step's action.

The LLM receives the current observation as a structured prompt and returns
a JSON tool call. A fallback heuristic is used if the model output cannot
be parsed.

STDOUT FORMAT (MANDATORY — do not modify):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables required
-------------------------------
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.env.environment import DisasterReliefEnv

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_KEY: str = os.environ.get("HF_TOKEN", "") or os.environ.get("API_KEY", "")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK: str = "disaster-relief-coordination"

if not API_KEY:
    print(
        "WARNING: HF_TOKEN is not set. API calls may fail.",
        file=sys.stderr,
    )

# OpenAI-compatible client (works with HF Inference, OpenAI, Together, etc.)
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
# LLM-based action selection
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a disaster relief coordination AI. You triage reports, dispatch resources, and resolve incidents under time pressure. Your score depends on RESOLVING reports — not just triaging them.

Reply with ONLY a JSON object: {"tool": "<name>", "args": {<arguments>}}

TOOLS (use only these):
1. call_intake_agent(report_id): Runs classify+urgency+verify in ONE call. Use this for ANY unclassified report. NEVER call classify_report, assess_report_urgency, or verify_report separately — intake does all three.
2. send_resource(resource_id, report_id): Dispatch a resource to handle a report. This is how reports get RESOLVED. You must dispatch to score points.
3. mark_false_report(report_id, reason): Flag a report with verification_confidence < 0.3 as false.
4. check_operation(target_id): Check status of a report or assignment.
5. call_monitor_agent(target_id): Monitor/close a case.
6. close_case(assignment_id): Close a completed assignment.
7. reroute_resource(resource_id): Reroute a stuck deployed resource.
8. get_resources(): List all resources.
9. call_dispatch_agent(resource_id, report_id): Like send_resource but checks inventory first.

CRITICAL RULES:
- RESOLVING reports is what scores points. Triage alone scores ZERO.
- After intake on a report, IMMEDIATELY dispatch a resource to it in the next step.
- Do NOT intake all reports before dispatching — interleave: intake → dispatch → intake → dispatch.
- CRITICAL reports (marked CRIT) with deadlines MUST be dispatched ASAP. Earlier = higher score multiplier.
- FLOOD ZONES (flood_depth >= 2): ONLY send flood-capable resources (rescue_boat, helicopter). Sending non-flood resources FAILS and wastes a step.
- Match resource types: flood→rescue_boat, medical→medical_team, structural_collapse/fire/road_blockage→engineering_crew, evacuation→supply_truck.
- Reports with verification_confidence < 0.3 after intake: mark_false_report immediately (saves resources).
- NEVER repeat the same action twice in a row (repeat penalty).
- Steps are LIMITED. Every wasted step means fewer reports resolved.

PRIORITY ORDER each step:
1. If a verified-false report exists (confidence < 0.3): mark_false_report
2. If a classified CRITICAL report has no resource assigned and a matching resource is available: send_resource
3. If an unclassified report exists: call_intake_agent on the highest-urgency one
4. If a classified non-critical report has no resource and a matching resource is available: send_resource
5. If assignments are stuck: reroute_resource
6. If nothing else: check_operation on an active assignment
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
        f"Step: {obs['step']}/{obs['max_steps']} ({obs['max_steps'] - obs['step']} remaining)\n\n"
        f"{summary}\n\n"
        "Return ONLY a JSON tool call. No explanation."
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
    """Build a concise text summary of the observation for the LLM (target <800 tokens)."""
    lines = []

    # Situation brief (only on step 0)
    if obs.get("situation_brief"):
        lines.append(obs["situation_brief"])
        lines.append("")

    # Zones — only show zones with notable conditions
    zones = obs.get("zones", [])
    zone_map = {z["id"]: z for z in zones}
    if zones:
        notable = [z for z in zones if z["flood_depth_level"] >= 1 or z["access_blocked"] or z["comms_blackout"] or z["severity"] >= 3]
        if notable:
            lines.append("Zone conditions:")
            for z in notable:
                flags = []
                if z["flood_depth_level"] >= 2:
                    flags.append(f"FLOOD-depth{z['flood_depth_level']}(needs boat/heli)")
                elif z["flood_depth_level"] == 1:
                    flags.append("shallow-flood")
                if z["access_blocked"]:
                    flags.append("BLOCKED")
                if z["comms_blackout"]:
                    flags.append("NO-COMMS")
                lines.append(f"  {z['id']}: sev={z['severity']} {' '.join(flags)}")

    # Resources — show BEFORE reports so LLM knows what's available for dispatch
    resources = obs.get("available_resources", [])
    avail = [r for r in resources if r["status"] == "available"]
    lines.append(f"\nResources: {len(avail)} avail / {len(resources)} total")
    for r in avail:
        flood = " FLOOD-OK" if r.get("can_traverse_flood") else ""
        lines.append(f"  {r['id']}: {r['type']}{flood}")

    # Pending reports — cap at 6, prioritized, with dispatch hints
    pending = obs.get("pending_reports", [])
    if pending:
        # Separate into needs-intake vs ready-to-dispatch
        needs_intake = [r for r in pending if not r["classified"]]
        ready = [r for r in pending if r["classified"] and not r.get("assigned_resource_id")]
        dispatched = [r for r in pending if r.get("assigned_resource_id")]

        if ready:
            lines.append(f"\nREADY TO DISPATCH ({len(ready)}):")
            for r in ready[:4]:
                crit = "CRIT " if r["is_critical"] else ""
                dl = f" DL=s{r['deadline_step']}" if r.get("deadline_step") else ""
                ver = ""
                if r["verified"] and r.get("verification_confidence") is not None:
                    conf = r["verification_confidence"]
                    ver = f" v={conf:.1f}"
                    if conf < 0.3:
                        ver += " FALSE?"
                zone = zone_map.get(r.get("zone_id", ""), {})
                flood_note = f" FLOOD-ZONE" if zone.get("flood_depth_level", 0) >= 2 else ""
                needed = _CATEGORY_RESOURCE.get(r.get("category", ""), "?")
                lines.append(f"  {r['id']}: {crit}{r['category']} z={r['zone_id']}{flood_note} u={r['urgency']}{dl}{ver} needs={needed}")

        if needs_intake:
            lines.append(f"\nNEEDS INTAKE ({len(needs_intake)}):")
            for r in needs_intake[:3]:
                crit = "CRIT " if r["is_critical"] else ""
                dl = f" DL=s{r['deadline_step']}" if r.get("deadline_step") else ""
                lines.append(f"  {r['id']}: {crit}unclassified z={r['zone_id']} u={r['urgency']}{dl}")

        if dispatched:
            lines.append(f"\nDISPATCHED ({len(dispatched)}): {', '.join(r['id'] for r in dispatched[:3])}")

    # Active assignments — compact
    assignments = obs.get("active_assignments", [])
    if assignments:
        lines.append(f"\nAssignments ({len(assignments)}):")
        for a in assignments[:4]:
            stuck = " STUCK" if a["stuck"] else ""
            lines.append(f"  {a['id']}: {a['resource_id']}→{a['report_id']} ETA=s{a['expected_completion_step']}{stuck}")

    # Warnings — always show
    for w in obs.get("warnings", []):
        lines.append(f"  ⚠ {w}")

    # Last feedback — truncated
    if obs.get("last_action_result"):
        lines.append(f"\nResult: {obs['last_action_result'][:120]}")
    if obs.get("last_action_error"):
        lines.append(f"\nError: {obs['last_action_error'][:120]}")

    return "\n".join(lines)


_CATEGORY_RESOURCE = {
    "flood": "rescue_boat",
    "structural_collapse": "engineering_crew",
    "medical": "medical_team",
    "fire": "engineering_crew",
    "road_blockage": "engineering_crew",
    "evacuation": "supply_truck",
}


def _pick_resource(
    rpt: Dict[str, Any],
    avail: List[Dict[str, Any]],
    zones: Dict[str, Any],
    allow_mismatch: bool = False,
) -> Optional[Dict[str, Any]]:
    """Pick the best available resource for a report (type-match + flood-aware).
    
    If allow_mismatch is False, returns None when no type-matched resource is found.
    Critical reports near deadline always allow mismatch.
    """
    zone = zones.get(rpt.get("zone_id", ""), {})
    flood_depth = zone.get("flood_depth_level", 0)

    candidates = list(avail)
    if flood_depth >= 2:
        flood_capable = [r for r in candidates if r.get("can_traverse_flood", False)]
        if flood_capable:
            candidates = flood_capable
        else:
            return None  # can't reach this zone

    # Prefer type match
    needed = _CATEGORY_RESOURCE.get(rpt.get("category", ""), "")
    for r in candidates:
        if r["type"] == needed:
            return r

    # Fallback: any available resource (only if allowed or critical with tight deadline)
    if allow_mismatch or rpt.get("is_critical"):
        return candidates[0] if candidates else None
    return None


def _heuristic_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic flood-aware, type-aware heuristic. Used as fallback when
    the LLM returns an unparseable response, or for offline testing.

    Optimized to maximize throughput in limited steps:
    1. Flag verified false reports immediately
    2. Dispatch critical triaged reports (type-matched)
    3. Intake unclassified reports
    4. Dispatch non-critical triaged reports
    5. Monitor only when nothing else to do
    """
    pending = obs.get("pending_reports", [])
    resources = obs.get("available_resources", [])
    assignments = obs.get("active_assignments", [])
    zones = {z["id"]: z for z in obs.get("zones", [])}

    avail = [r for r in resources if r["status"] == "available"]

    # Focus on actionable reports (not yet dispatched)
    actionable = [
        r for r in pending
        if r["status"] in ("pending", "triaged")
        and r.get("assigned_resource_id") is None
    ]

    if not actionable:
        if assignments:
            stuck = [a for a in assignments if a.get("stuck")]
            if stuck:
                return {"tool": "reroute_resource", "args": {"resource_id": stuck[0]["resource_id"]}}
            return {"tool": "check_operation", "args": {"target_id": assignments[0]["report_id"]}}
        return {"tool": "get_resources", "args": {}}

    # Priority 1: Flag verified low-confidence reports as false (saves resources)
    for rpt in actionable:
        if rpt["verified"] and rpt.get("verification_confidence", 1.0) < 0.3:
            return {"tool": "mark_false_report", "args": {
                "report_id": rpt["id"], "reason": "low verification confidence"
            }}

    # Priority 2: Dispatch critical triaged reports with type-matched resources
    for rpt in actionable:
        if rpt["classified"] and rpt["is_critical"] and avail:
            best = _pick_resource(rpt, avail, zones)
            if best:
                return {"tool": "send_resource", "args": {
                    "resource_id": best["id"], "report_id": rpt["id"]
                }}

    # Priority 3: Intake unclassified reports (critical-flagged first)
    for rpt in actionable:
        if not rpt["classified"]:
            return {"tool": "call_intake_agent", "args": {"report_id": rpt["id"]}}

    # Priority 4: Dispatch non-critical triaged reports
    for rpt in actionable:
        if rpt["classified"] and avail:
            best = _pick_resource(rpt, avail, zones)
            if best:
                return {"tool": "send_resource", "args": {
                    "resource_id": best["id"], "report_id": rpt["id"]
                }}

    # Priority 5: Monitor active assignments
    if assignments:
        stuck = [a for a in assignments if a.get("stuck")]
        if stuck:
            return {"tool": "reroute_resource", "args": {"resource_id": stuck[0]["resource_id"]}}
        return {"tool": "check_operation", "args": {"target_id": assignments[0]["report_id"]}}

    return {"tool": "get_resources", "args": {}}


# ---------------------------------------------------------------------------
# Helper: structured stdout logging (MANDATORY format)
# ---------------------------------------------------------------------------


def log_start(task_name: str) -> None:
    print(
        f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}",
        flush=True,
    )


def log_step(
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    action_str = f"{action.get('tool', 'unknown')}({json.dumps(action.get('args', {}), separators=(',', ':'))})"
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
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
    result = env.reset(task_name, seed=seed)
    obs = result["observation"]

    log_start(task_name)

    done = False
    step = 0
    rewards: List[float] = []
    success = False
    grade: Dict[str, Any] = {}

    try:
        while not done:
            if use_llm:
                action = get_llm_action(obs, task_name)
            else:
                action = _heuristic_action(obs)

            result = env.step(action)
            next_obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            error = obs.get("last_action_error")

            rewards.append(reward)

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=done,
                error=error,
            )

            obs = next_obs
            step += 1

        grade = env.grade()
        score = min(max(grade["score"], 0.0), 1.0)  # clamp to [0, 1]
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr)
        score = 0.0
        success = False

    finally:
        env.close()
        log_end(success=success, steps=step, score=score, rewards=rewards)

    return {
        "task": task_name,
        "score": score,
        "total_reward": sum(rewards),
        "critical_missed": grade.get("critical_missed", 0),
        "reports_resolved": grade.get("reports_resolved", 0),
        "total_reports": grade.get("total_reports", 0),
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

    print("\n" + "=" * 60, file=sys.stderr, flush=True)
    print("SUMMARY", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)
    for r in results:
        print(
            f"  {r['task']}: score={r['score']:.4f}  "
            f"reward={r['total_reward']:.1f}  "
            f"resolved={r['reports_resolved']}/{r['total_reports']}  "
            f"critical_missed={r['critical_missed']}  "
            f"steps={r['steps']}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
