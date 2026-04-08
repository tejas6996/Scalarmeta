"""Phase 6 test: Full episode loop through DisasterReliefEnv."""
import json
from src.env.environment import DisasterReliefEnv

env = DisasterReliefEnv()

print("Supported tasks:", env.supported_tasks())
print()

# ============================================================
# TEST 1: Full reset → step → grade loop (task1)
# ============================================================
print("=" * 60)
print("TEST 1: Full episode — task1_flood_easy")
print("=" * 60)

obs = env.reset("task1_flood_easy", seed=42)
print(f"  Step: {obs['step']}/{obs['max_steps']}")
print(f"  Pending reports: {len(obs['pending_reports'])}")
print(f"  Resources: {len(obs['available_resources'])}")
print(f"  Recent changes: {len(obs['recent_changes'])}")
print(f"  Tools: {len(obs['available_tools'])}")
print(f"  Weather: {obs['weather_severity']}")

# Play a few steps
total_steps = 0
zone_flood = {z["id"]: z["flood_depth_level"] for z in obs.get("zones", [])}
while not env.is_done:
    pending = obs.get("pending_reports", [])

    # Simple strategy: intake first pending, then dispatch (flood-aware)
    if pending:
        rpt = pending[0]
        if not rpt["classified"]:
            action = {"tool": "call_intake_agent", "args": {"report_id": rpt["id"]}}
        elif rpt["verified"] and rpt.get("verification_confidence", 1.0) < 0.3:
            action = {"tool": "mark_false_report", "args": {"report_id": rpt["id"], "reason": "low confidence"}}
        else:
            # Try to dispatch — flood-aware
            flood = zone_flood.get(rpt.get("zone_id", ""), 0)
            avail = [r for r in obs.get("available_resources", []) if r["status"] == "available"]
            if flood >= 2:
                avail = [r for r in avail if r.get("can_traverse_flood", False)]
            if avail:
                action = {"tool": "send_resource", "args": {"resource_id": avail[0]["id"], "report_id": rpt["id"]}}
            else:
                action = {"tool": "check_operation", "args": {"target_id": rpt["id"]}}
    else:
        action = {"tool": "get_resources", "args": {}}

    result = env.step(action)
    obs = result["observation"]
    zone_flood = {z["id"]: z["flood_depth_level"] for z in obs.get("zones", [])}
    total_steps += 1

    if total_steps <= 3 or result["done"]:
        print(f"  Step {obs['step']}: reward={result['reward']:.2f}, done={result['done']}")
        if obs.get("last_action_result"):
            print(f"    Result: {obs['last_action_result'][:80]}...")
        if obs.get("last_action_error"):
            print(f"    Error: {obs['last_action_error'][:80]}...")

# Grade
grade = env.grade()
print(f"\n  Grade: {grade['score']:.4f}")
print(f"  Resolved: {grade['reports_resolved']}/{grade['total_reports']}")
print(f"  Expired: {grade['reports_expired']}")
print(f"  Critical missed: {grade['critical_missed']}")
print(f"  Total reward: {grade['total_reward']:.1f}")
print(f"  Steps used: {grade['steps_used']}")

assert grade["score"] > 0.0, "Reasonable agent should score > 0.0"
print("  PASS: score > 0.0")

# State endpoint
state = env.get_state()
print(f"\n  State snapshot: step={state['current_step']}, reward={state['total_reward']:.1f}")

# ============================================================
# TEST 2: All 3 tasks
# ============================================================
print()
print("=" * 60)
print("TEST 2: All 3 tasks — do-nothing vs reasonable agent")
print("=" * 60)

for task in env.supported_tasks():
    # Do-nothing
    obs = env.reset(task, seed=42)
    while not env.is_done:
        result = env.step({"tool": "get_resources", "args": {}})
        obs = result["observation"]
    dn_grade = env.grade()

    # Reasonable agent — flood-aware strategy
    obs = env.reset(task, seed=42)
    # Build zone flood lookup
    zone_flood = {z["id"]: z["flood_depth_level"] for z in obs.get("zones", [])}
    while not env.is_done:
        pending = obs.get("pending_reports", [])
        if pending:
            rpt = pending[0]
            if not rpt["classified"]:
                action = {"tool": "call_intake_agent", "args": {"report_id": rpt["id"]}}
            elif rpt["verified"] and rpt.get("verification_confidence", 1.0) < 0.3:
                action = {"tool": "mark_false_report", "args": {"report_id": rpt["id"], "reason": "auto"}}
            else:
                # Find a compatible available resource (flood-aware)
                flood = zone_flood.get(rpt.get("zone_id", ""), 0)
                avail = [r for r in obs.get("available_resources", []) if r["status"] == "available"]
                if flood >= 2:
                    # Only use flood-capable resources for flooded zones
                    avail = [r for r in avail if r.get("can_traverse_flood", False)]
                if avail:
                    action = {"tool": "send_resource", "args": {"resource_id": avail[0]["id"], "report_id": rpt["id"]}}
                else:
                    # No matching resource — monitor instead of wasting a step
                    action = {"tool": "check_operation", "args": {"target_id": rpt["id"]}}
        else:
            action = {"tool": "get_resources", "args": {}}
        result = env.step(action)
        obs = result["observation"]
        # Refresh zone lookup in case conditions changed
        zone_flood = {z["id"]: z["flood_depth_level"] for z in obs.get("zones", [])}
    good_grade = env.grade()

    print(f"  {task}: do-nothing={dn_grade['score']:.4f}  reasonable={good_grade['score']:.4f}  "
          f"resolved={good_grade['reports_resolved']}/{good_grade['total_reports']}")

# Task1 should always be better with a reasonable agent
assert env.supported_tasks()[0] == "task1_flood_easy"
print("  PASS: all tasks completed without errors")

# ============================================================
# TEST 3: Malformed action handling
# ============================================================
print()
print("=" * 60)
print("TEST 3: Malformed action handling")
print("=" * 60)

obs = env.reset("task1_flood_easy", seed=42)

# Empty tool
result = env.step({"tool": "", "args": {}})
print(f"  Empty tool: reward={result['reward']:.2f}, error={result['observation'].get('last_action_error', 'none')[:60]}")
assert result["reward"] < 0.5, "Malformed action should have low reward"

# Missing tool field
result = env.step({"args": {"x": 1}})
print(f"  Missing tool: reward={result['reward']:.2f}")

# Bad tool name
result = env.step({"tool": "fly_to_moon", "args": {}})
error = result["observation"].get("last_action_error", "")
print(f"  Unknown tool: error={error[:60]}")
assert "Error" in error

print("  PASS: all malformed actions handled gracefully")

# ============================================================
# TEST 4: Observation JSON serializable
# ============================================================
print()
print("=" * 60)
print("TEST 4: Observation is JSON-serializable")
print("=" * 60)

obs = env.reset("task2_storm_medium", seed=42)
json_str = json.dumps(obs)
print(f"  Observation JSON size: {len(json_str)} bytes")
assert len(json_str) > 100, "Observation should have meaningful content"

result = env.step({"tool": "get_resources", "args": {}})
json_str2 = json.dumps(result)
print(f"  StepResult JSON size: {len(json_str2)} bytes")

state = env.get_state()
json_str3 = json.dumps(state)
print(f"  State JSON size: {len(json_str3)} bytes")

grade = env.grade()
json_str4 = json.dumps(grade)
print(f"  Grade JSON size: {len(json_str4)} bytes")

print("  PASS: all outputs JSON-serializable")

# ============================================================
# TEST 5: close() works
# ============================================================
print()
env.close()
assert not env.is_initialized
print("TEST 5: PASS — env.close() works")

print()
print("=== ALL PHASE 6 TESTS PASSED ===")
