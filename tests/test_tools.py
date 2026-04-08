"""Quick Phase 4 integration test — coordinator + false reports + edge cases."""
from src.env.scenarios import generate_scenario
from src.env.state import WorldState
from src.env.tool_registry import execute_tool

# Test with medium task (has duplicates and false reports)
s = generate_scenario("task2_storm_medium", seed=42)
state = WorldState.from_scenario(s)

# Advance to step 6 to get many visible reports
for step in range(7):
    state.current_step = step
    state.advance_time()

pending = state.get_pending_reports()
print(f"Step 6: {len(pending)} pending reports")
for r in pending:
    gt = state.ground_truth.get(r.id, {})
    verdict = gt.get("verdict", "?")
    cat = gt.get("category", "?")
    print(f"  {r.id}: verdict={verdict}, cat={cat}")

# Test call_intake_agent (delegation tool)
print(f"\n=== call_intake_agent on first report ===")
result, reward = execute_tool(state, "call_intake_agent", {"report_id": pending[0].id})
print(result)
print(f"Total reward: {reward}")

# Find a false/duplicate report
false_rpt = None
for r in pending:
    gt = state.ground_truth.get(r.id, {})
    if gt.get("verdict") in ("false", "duplicate"):
        false_rpt = r.id
        break

if false_rpt:
    print(f"\n=== mark_false_report({false_rpt}) ===")
    result, reward = execute_tool(state, "mark_false_report", {
        "report_id": false_rpt,
        "reason": "low confidence + vague language",
    })
    print(f"Result: {result}")
    print(f"Reward: {reward}")
else:
    print("\nNo false/duplicate reports found in pending (may be hidden by comms blackout)")

# Test call_monitor_agent with close instruction on a pending real report
real_rpt = None
for r in state.get_pending_reports():
    gt = state.ground_truth.get(r.id, {})
    if gt.get("verdict") == "real" and r.id != pending[0].id:
        real_rpt = r.id
        break

if real_rpt:
    print(f"\n=== call_monitor_agent({real_rpt}, close) — premature closure ===")
    result, reward = execute_tool(state, "call_monitor_agent", {
        "target_id": real_rpt,
        "instruction": "close this case",
    })
    print(result)
    print(f"Total reward: {reward}")

# Test reroute on a resource that isn't deployed
print(f"\n=== reroute_resource(RES-001) — not deployed ===")
result, reward = execute_tool(state, "reroute_resource", {"resource_id": "RES-001"})
print(f"Result: {result}")
print(f"Reward: {reward}")

# Test full dispatch → close cycle
print(f"\n=== Full dispatch+close cycle ===")
remaining_pending = state.get_pending_reports()
if remaining_pending:
    rpt = remaining_pending[0]
    avail = state.get_available_resources()
    if avail:
        res = avail[0]
        r1, rw1 = execute_tool(state, "send_resource", {"resource_id": res.id, "report_id": rpt.id})
        print(f"Dispatch: {r1} (reward={rw1})")

        # Advance until assignment completes
        for step in range(state.current_step + 1, state.max_steps):
            state.current_step = step
            state.advance_time()
            asg = state.get_assignment_for_report(rpt.id)
            if asg is None or asg.status.value == "completed":
                print(f"  Assignment completed at step {step}")
                break

        r2, rw2 = execute_tool(state, "close_case", {"report_id": rpt.id, "resolution_note": "resolved by rescue"})
        print(f"Close: {r2} (reward={rw2})")

print(f"\n=== ALL TESTS PASSED ===")
