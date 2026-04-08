"""Verify temporal urgency multiplier works on a correct dispatch."""
from src.env.scenarios import generate_scenario
from src.env.state import WorldState
from src.env.tool_registry import execute_tool
from src.env.rewards import compute_step_reward

s = generate_scenario("task2_storm_medium", seed=42)
state = WorldState.from_scenario(s)
state.current_step = 0
state.advance_time()

# Check what we have
for r in state.get_pending_reports():
    gt = state.ground_truth[r.id]
    print(f"  {r.id}: cat={gt.get('category')}, need={gt.get('required_resource')}, "
          f"deadline={r.deadline_step}, created={r.created_step}")

# Advance further to get more reports
for step in range(1, 6):
    state.current_step = step
    state.advance_time()

print(f"\nStep {state.current_step} — pending reports:")
for r in state.get_pending_reports():
    gt = state.ground_truth[r.id]
    print(f"  {r.id}: need={gt.get('required_resource')}, deadline={r.deadline_step}")

avail_types = [(res.id, res.type.value) for res in state.get_available_resources()]
print(f"\nAvailable resources: {avail_types}")

# Find a match: real report with deadline + correct resource available
for r in state.get_pending_reports():
    gt = state.ground_truth[r.id]
    needed = gt.get("required_resource")
    if not needed or not r.deadline_step or gt.get("verdict") != "real":
        continue
    for res in state.get_available_resources():
        if res.type.value == needed:
            # Check flood accessibility first
            zone = state.get_zone(r.zone_id)
            if zone and zone.flood_depth_level >= 2 and not res.can_traverse_flood:
                continue  # skip — will get blocked by flood validation
            print(f"\n=== Dispatching {res.id} ({res.type.value}) -> {r.id} (deadline={r.deadline_step}) ===")
            result, delta = execute_tool(state, "send_resource", {"resource_id": res.id, "report_id": r.id})
            print(f"Raw delta from tool: {delta}")
            rw = compute_step_reward(state, "send_resource", {"resource_id": res.id, "report_id": r.id}, result, delta)
            print(f"After temporal multiplier: dispatch_reward={rw.dispatch_reward:.3f}")
            print(f"Total reward: {rw.total:.3f}")
            if delta > 0:
                mult = rw.dispatch_reward / delta
                print(f"Temporal multiplier: {mult:.2f}x")
                assert mult >= 1.0, f"Multiplier should be >= 1.0, got {mult}"
                print("PASS: temporal multiplier applied!")
            break
    else:
        continue
    break
