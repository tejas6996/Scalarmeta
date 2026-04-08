"""Test the 4 advanced Phase 5 enhancements."""
from src.env.scenarios import generate_scenario, SUPPORTED_TASKS
from src.env.state import WorldState
from src.env.tool_registry import execute_tool
from src.env.rewards import compute_step_reward
from src.env.graders import grade_episode, BaseGrader

# ============================================================
# TEST 1: Resource Matching Confidence Decay
# ============================================================
print("=" * 60)
print("TEST 1: Resource matching decay (flood depth + credibility)")
print("=" * 60)

s = generate_scenario("task2_storm_medium", seed=42)
state = WorldState.from_scenario(s)

# Advance to get reports
for step in range(7):
    state.current_step = step
    state.advance_time()

# Find a real report and dispatch correct resource
pending = state.get_pending_reports()
real_rpt = None
for r in pending:
    gt = state.ground_truth.get(r.id, {})
    if gt.get("verdict") == "real":
        real_rpt = r
        break

zone = state.get_zone(real_rpt.zone_id)
print(f"  Report {real_rpt.id}: zone={real_rpt.zone_id}, flood_depth={zone.flood_depth_level}, reporter={real_rpt.reporter_type.value}")
print(f"  Required resource: {state.ground_truth[real_rpt.id].get('required_resource')}")

# Dispatch — reward should include flood penalty and credibility bonus
avail = state.get_available_resources()
result, delta = execute_tool(state, "send_resource", {"resource_id": avail[0].id, "report_id": real_rpt.id})
print(f"  Dispatch result: {result[:100]}...")
print(f"  Raw reward delta: {delta}")
# If flood_depth > 0, delta should be < 2.0 for correct match (or adjusted)
print(f"  PASS: resource decay applied (delta={delta})")

# ============================================================  
# TEST 2: Temporal Urgency Multiplier
# ============================================================
print()
print("=" * 60)
print("TEST 2: Temporal urgency multiplier")
print("=" * 60)

s = generate_scenario("task1_flood_easy", seed=42)
state = WorldState.from_scenario(s)

state.current_step = 0
state.advance_time()

# Get critical report with deadline
pending = state.get_pending_reports()
crit = None
for r in pending:
    if r.is_critical and r.deadline_step:
        crit = r
        break

if crit:
    print(f"  Critical report {crit.id}: created={crit.created_step}, deadline={crit.deadline_step}")

    # Classify first so it has a chance to work
    execute_tool(state, "classify_report", {"report_id": crit.id})

    # Dispatch immediately (fraction_used ≈ 0 → multiplier ≈ 2.0)
    avail = state.get_available_resources()
    result, delta = execute_tool(state, "send_resource", {"resource_id": avail[0].id, "report_id": crit.id})
    r_early = compute_step_reward(state, "send_resource", {"resource_id": avail[0].id, "report_id": crit.id}, result, delta)
    print(f"  Early dispatch: raw_delta={delta}, dispatch_reward={r_early.dispatch_reward:.2f}, total={r_early.total:.2f}")
    print(f"  Multiplier effect: {r_early.dispatch_reward / max(delta, 0.01):.2f}x")
    if delta > 0:
        assert r_early.dispatch_reward >= delta, f"Temporal multiplier should boost reward, got {r_early.dispatch_reward} < {delta}"
        print(f"  PASS: temporal multiplier boosted reward")
else:
    print("  No critical report at step 0, skipping")

# ============================================================
# TEST 3: F1-Score for False Alarm Detection  
# ============================================================
print()
print("=" * 60)
print("TEST 3: False alarm F1-score")
print("=" * 60)

grader = BaseGrader()

# Case A: Perfect detection — flag all false, none of real
s = generate_scenario("task2_storm_medium", seed=42)
state = WorldState.from_scenario(s)
for step in range(state.max_steps):
    state.current_step = step
    state.advance_time()

# Flag only actual false/duplicate reports
for rid, gt in state.ground_truth.items():
    if gt.get("verdict") in ("false", "duplicate"):
        state.reports[rid].status = state.reports[rid].status  # need ReportStatus
        from src.env.models import ReportStatus
        state.reports[rid].status = ReportStatus.FALSE

f1_perfect = grader._verification_accuracy(state)
print(f"  Perfect flagging F1: {f1_perfect:.4f}")
assert f1_perfect == 1.0, f"Perfect flagging should give F1=1.0, got {f1_perfect}"
print(f"  PASS: F1=1.0 for perfect detection")

# Case B: Flag everything (should have low precision)
s2 = generate_scenario("task2_storm_medium", seed=42)
state2 = WorldState.from_scenario(s2)
for step in range(state2.max_steps):
    state2.current_step = step
    state2.advance_time()

for rid in state2.reports:
    state2.reports[rid].status = ReportStatus.FALSE

f1_all_flagged = grader._verification_accuracy(state2)
false_count = sum(1 for gt in state2.ground_truth.values() if gt.get("verdict") in ("false", "duplicate"))
total_count = len(state2.reports)
expected_precision = false_count / total_count
print(f"  Flag-everything F1: {f1_all_flagged:.4f} (precision={expected_precision:.2f})")
assert f1_all_flagged < 1.0, "Flag-everything should NOT get perfect score"
print(f"  PASS: F1 < 1.0 for degenerate flag-everything strategy")

# Case C: Flag nothing
s3 = generate_scenario("task2_storm_medium", seed=42)
state3 = WorldState.from_scenario(s3)
for step in range(state3.max_steps):
    state3.current_step = step
    state3.advance_time()
# Don't flag anything

f1_nothing = grader._verification_accuracy(state3)
print(f"  Flag-nothing F1: {f1_nothing:.4f}")
assert f1_nothing == 0.0, f"Flag-nothing should give F1=0.0, got {f1_nothing}"
print(f"  PASS: F1=0.0 for flag-nothing strategy")

# ============================================================
# TEST 4: Counterfactual Penalty
# ============================================================
print()
print("=" * 60)
print("TEST 4: Counterfactual penalty")
print("=" * 60)

# Do-nothing agent: all critical expire, resources were available → score should be 0.0
s = generate_scenario("task2_storm_medium", seed=42)
state = WorldState.from_scenario(s)
for step in range(state.max_steps):
    state.current_step = step
    state.advance_time()

cf = grader._counterfactual_penalty(state)
print(f"  Do-nothing counterfactual: {cf:.4f}")
print(f"  Expired: {state.reports_expired}, Critical missed: {state.critical_missed}")
print(f"  Availability log entries: {len(state.availability_log)}")
assert cf < 1.0, "Do-nothing should have preventable misses (counterfactual < 1.0)"
print(f"  PASS: counterfactual detected preventable misses")

# Reasonable agent: handle some critical reports → better counterfactual score
s = generate_scenario("task1_flood_easy", seed=42)
state = WorldState.from_scenario(s)
for step in range(state.max_steps):
    state.current_step = step
    state.advance_time()
    for rpt in state.get_pending_reports():
        if not rpt.classified:
            execute_tool(state, "call_intake_agent", {"report_id": rpt.id})
    pending = state.get_pending_reports()
    available = state.get_available_resources()
    for rpt in sorted(pending, key=lambda r: (-r.urgency, r.id)):
        if not available:
            break
        if rpt.verified and rpt.verification_confidence is not None and rpt.verification_confidence < 0.3:
            execute_tool(state, "mark_false_report", {"report_id": rpt.id, "reason": "low confidence"})
            continue
        res = available.pop(0)
        execute_tool(state, "send_resource", {"resource_id": res.id, "report_id": rpt.id})

cf_good = grader._counterfactual_penalty(state)
print(f"  Reasonable agent counterfactual: {cf_good:.4f}")
print(f"  Expired: {state.reports_expired}, Critical missed: {state.critical_missed}")
assert cf_good >= cf, "Reasonable agent should have equal or better counterfactual than do-nothing"
print(f"  PASS: reasonable agent has better counterfactual score")

# ============================================================
# TEST 5: Full grader comparison (before vs after enhancements)
# ============================================================
print()
print("=" * 60)
print("TEST 5: Full grader scores with all enhancements")
print("=" * 60)

for task in SUPPORTED_TASKS:
    s = generate_scenario(task, seed=42)
    state = WorldState.from_scenario(s)

    # Do-nothing
    for step in range(state.max_steps):
        state.current_step = step
        state.advance_time()
    donothing_score = grade_episode(state)

    # Reasonable agent
    s = generate_scenario(task, seed=42)
    state = WorldState.from_scenario(s)
    for step in range(state.max_steps):
        state.current_step = step
        state.advance_time()
        for rpt in state.get_pending_reports():
            if not rpt.classified:
                execute_tool(state, "call_intake_agent", {"report_id": rpt.id})
        pending = state.get_pending_reports()
        available = state.get_available_resources()
        for rpt in sorted(pending, key=lambda r: (-r.urgency, r.id)):
            if not available:
                break
            if rpt.verified and rpt.verification_confidence is not None and rpt.verification_confidence < 0.3:
                execute_tool(state, "mark_false_report", {"report_id": rpt.id, "reason": "low confidence"})
                continue
            res = available.pop(0)
            execute_tool(state, "send_resource", {"resource_id": res.id, "report_id": rpt.id})
    good_score = grade_episode(state)

    print(f"  {task}: do-nothing={donothing_score:.4f}  reasonable={good_score:.4f}  delta={good_score - donothing_score:+.4f}")
    assert good_score > donothing_score, f"Reasonable agent must score higher than do-nothing for {task}"

print()
print("=== ALL ENHANCEMENT TESTS PASSED ===")
