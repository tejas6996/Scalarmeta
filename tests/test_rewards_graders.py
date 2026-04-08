"""Phase 5 test: rewards + graders for all 3 tasks."""
from src.env.scenarios import generate_scenario, SUPPORTED_TASKS
from src.env.state import WorldState
from src.env.tool_registry import execute_tool
from src.env.rewards import compute_step_reward, compute_no_action_reward
from src.env.graders import grade_episode

print("=" * 60)
print("TEST 1: Do-nothing agent (all 3 tasks) — should score ~0.0")
print("=" * 60)

for task in SUPPORTED_TASKS:
    s = generate_scenario(task, seed=42)
    state = WorldState.from_scenario(s)

    for step in range(state.max_steps):
        state.current_step = step
        state.advance_time()
        compute_no_action_reward(state)

    score = grade_episode(state)
    print(f"  {task}: score={score:.4f}  reward={state.total_reward:.1f}  "
          f"resolved={state.reports_resolved}  expired={state.reports_expired}")

print()
print("=" * 60)
print("TEST 2: Reasonable agent on task1 — intake + dispatch everything")
print("=" * 60)

s = generate_scenario("task1_flood_easy", seed=42)
state = WorldState.from_scenario(s)

for step in range(state.max_steps):
    state.current_step = step
    state.advance_time()

    # Process pending reports
    pending = state.get_pending_reports()
    for rpt in pending:
        if not rpt.classified:
            result, delta = execute_tool(state, "call_intake_agent", {"report_id": rpt.id})
            r = compute_step_reward(state, "call_intake_agent", {"report_id": rpt.id}, result, delta)

    # Dispatch available resources to triaged reports
    pending = state.get_pending_reports()
    available = state.get_available_resources()
    for rpt in sorted(pending, key=lambda r: (-r.urgency, r.id)):
        if not available:
            break
        if rpt.verified and rpt.verification_confidence is not None and rpt.verification_confidence < 0.3:
            # Skip suspicious reports — mark as false
            result, delta = execute_tool(state, "mark_false_report", {
                "report_id": rpt.id, "reason": "low confidence"
            })
            compute_step_reward(state, "mark_false_report", {"report_id": rpt.id}, result, delta)
            continue

        res = available.pop(0)
        result, delta = execute_tool(state, "send_resource", {
            "resource_id": res.id, "report_id": rpt.id
        })
        compute_step_reward(state, "send_resource", {"resource_id": res.id, "report_id": rpt.id}, result, delta)

score = grade_episode(state)
print(f"  Score: {score:.4f}")
print(f"  Total reward: {state.total_reward:.1f}")
print(f"  Resolved: {state.reports_resolved}/{state.total_reports}")
print(f"  Expired: {state.reports_expired}")
print(f"  Critical missed: {state.critical_missed}")
print(f"  False flagged: {state.reports_false_flagged}")
print(f"  Memory entries: {len(state.memory)}")

# Verify score > 0.0 for reasonable agent
assert score > 0.0, f"Reasonable agent should score > 0.0 but got {score}"
print(f"  PASS: score > 0.0")

print()
print("=" * 60)
print("TEST 3: Reward structure verification")
print("=" * 60)

s = generate_scenario("task1_flood_easy", seed=42)
state = WorldState.from_scenario(s)
state.current_step = 0
state.advance_time()

pending = state.get_pending_reports()
rpt_id = pending[0].id

# Good action: classify
result, delta = execute_tool(state, "classify_report", {"report_id": rpt_id})
r = compute_step_reward(state, "classify_report", {"report_id": rpt_id}, result, delta)
print(f"  Classify: base={r.base} triage={r.triage_reward} penalty={r.penalty} total={r.total}")
assert r.base == 0.5, "Base should be 0.5"
assert r.triage_reward > 0, "Correct classify should have positive triage reward"

# Repeat same action — should get penalty
result, delta = execute_tool(state, "classify_report", {"report_id": rpt_id})
r = compute_step_reward(state, "classify_report", {"report_id": rpt_id}, result, delta)
print(f"  Repeat:   base={r.base} triage={r.triage_reward} penalty={r.penalty} total={r.total}")
assert r.penalty <= -0.5, "Repeat should have -0.5 penalty"
print(f"  PASS: repeat penalty detected")

# No-action reward
r_none = compute_no_action_reward(state)
print(f"  No-action: base={r_none.base} penalty={r_none.penalty} total={r_none.total}")
assert r_none.total == 0.0, f"No-action should be 0.0 (base 0.5 - penalty 0.5) but got {r_none.total}"
print(f"  PASS: no-action reward correct")

# Memory entries logged
print(f"  Memory entries: {len(state.memory)}")
assert len(state.memory) >= 2, "Should have at least 2 memory entries"
print(f"  PASS: memory logged")

print()
print("=" * 60)
print("TEST 4: Full episode + grade on all 3 tasks (reasonable agent)")
print("=" * 60)

for task in SUPPORTED_TASKS:
    s = generate_scenario(task, seed=42)
    state = WorldState.from_scenario(s)

    for step in range(state.max_steps):
        state.current_step = step
        state.advance_time()

        for rpt in state.get_pending_reports():
            if not rpt.classified:
                result, delta = execute_tool(state, "call_intake_agent", {"report_id": rpt.id})
                compute_step_reward(state, "call_intake_agent", {"report_id": rpt.id}, result, delta)

        pending = state.get_pending_reports()
        available = state.get_available_resources()
        for rpt in sorted(pending, key=lambda r: (-r.urgency, r.id)):
            if not available:
                break
            if rpt.verified and rpt.verification_confidence is not None and rpt.verification_confidence < 0.3:
                result, delta = execute_tool(state, "mark_false_report", {
                    "report_id": rpt.id, "reason": "low confidence"
                })
                compute_step_reward(state, "mark_false_report", {"report_id": rpt.id}, result, delta)
                continue
            res = available.pop(0)
            result, delta = execute_tool(state, "send_resource", {
                "resource_id": res.id, "report_id": rpt.id
            })
            compute_step_reward(state, "send_resource", {"resource_id": res.id, "report_id": rpt.id}, result, delta)

    score = grade_episode(state)
    print(f"  {task}: score={score:.4f}  reward={state.total_reward:.1f}  "
          f"resolved={state.reports_resolved}/{state.total_reports}  "
          f"expired={state.reports_expired}  flagged={state.reports_false_flagged}")

print()
print("=== ALL PHASE 5 TESTS PASSED ===")
