"""Focused unit test for temporal urgency multiplier math."""
from src.env.scenarios import generate_scenario
from src.env.state import WorldState
from src.env.models import ReportStatus
from src.env.rewards import compute_step_reward

s = generate_scenario("task1_flood_easy", seed=42)
state = WorldState.from_scenario(s)
state.current_step = 0
state.advance_time()

# RPT-001 is critical with deadline=9, created=0
rpt = state.reports["RPT-001"]
print(f"Report: {rpt.id}, deadline={rpt.deadline_step}, created={rpt.created_step}")
print(f"Window: {rpt.deadline_step - rpt.created_step} steps")

# Simulate a successful dispatch (positive delta) at step 0
# fraction_used = (0 - 0) / (9 - 0) = 0.0
# multiplier = 1.0 + (1.0 - 0.0) = 2.0
r1 = compute_step_reward(state, "send_resource", {"resource_id": "RES-002", "report_id": "RPT-001"}, "OK", 2.0)
print(f"\nStep 0 dispatch (fraction_used=0.0):")
print(f"  Raw delta: 2.0, After multiplier: {r1.dispatch_reward:.3f}")
print(f"  Multiplier: {r1.dispatch_reward / 2.0:.2f}x")
assert abs(r1.dispatch_reward - 4.0) < 0.01, f"Expected 4.0 (2.0x), got {r1.dispatch_reward}"
print(f"  PASS: 2.0x multiplier at step 0")

# Simulate at step 6 (fraction_used = 6/9 ≈ 0.667)
# multiplier = 1.0 + (1.0 - 0.667) = 1.333
state2 = WorldState.from_scenario(s)
state2.current_step = 6
for step in range(7):
    state2.current_step = step
    state2.advance_time()

r2 = compute_step_reward(state2, "send_resource", {"resource_id": "RES-002", "report_id": "RPT-001"}, "OK", 2.0)
expected_mult = 1.0 + (1.0 - 6.0/9.0)
print(f"\nStep 6 dispatch (fraction_used={6/9:.3f}):")
print(f"  Raw delta: 2.0, After multiplier: {r2.dispatch_reward:.3f}")
print(f"  Expected multiplier: {expected_mult:.3f}x")
assert abs(r2.dispatch_reward - 2.0 * expected_mult) < 0.01, f"Expected {2.0*expected_mult:.3f}, got {r2.dispatch_reward}"
print(f"  PASS: {expected_mult:.2f}x multiplier at step 6")

# Simulate at step 8 (fraction_used = 8/9 ≈ 0.889)
# multiplier = 1.0 + (1.0 - 0.889) = 1.111
state3 = WorldState.from_scenario(s)
for step in range(9):
    state3.current_step = step
    state3.advance_time()

r3 = compute_step_reward(state3, "send_resource", {"resource_id": "RES-002", "report_id": "RPT-001"}, "OK", 2.0)
expected_mult3 = 1.0 + (1.0 - 8.0/9.0)
print(f"\nStep 8 dispatch (fraction_used={8/9:.3f}):")
print(f"  Raw delta: 2.0, After multiplier: {r3.dispatch_reward:.3f}")
print(f"  Expected multiplier: {expected_mult3:.3f}x")
assert abs(r3.dispatch_reward - 2.0 * expected_mult3) < 0.1, f"Expected ~{2.0*expected_mult3:.3f}, got {r3.dispatch_reward}"
print(f"  PASS: {expected_mult3:.2f}x multiplier at step 8")

# Verify monotonic: earlier dispatch = higher reward
assert r1.dispatch_reward > r2.dispatch_reward > r3.dispatch_reward
print(f"\nMonotonic: step0({r1.dispatch_reward:.2f}) > step6({r2.dispatch_reward:.2f}) > step8({r3.dispatch_reward:.2f})")
print("PASS: temporal multiplier is monotonically decreasing with time")

print("\n=== TEMPORAL MULTIPLIER TESTS PASSED ===")
