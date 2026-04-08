"""Phase 8 test: inference.py heuristic mode (no LLM API needed)."""
import sys
from io import StringIO

# We test the heuristic path only — no API key needed
from inference import _heuristic_action, _summarize_observation, run_task

# ============================================================
# TEST 1: Heuristic action selection
# ============================================================
print("=" * 60)
print("TEST 1: Heuristic action — unclassified report")
print("=" * 60)

obs = {
    "step": 0,
    "max_steps": 12,
    "task_name": "task1_flood_easy",
    "pending_reports": [
        {
            "id": "RPT-001",
            "raw_text": "Flooding at Market District",
            "zone_id": "ZONE-A",
            "category": "unknown",
            "urgency": 0,
            "status": "pending",
            "is_critical": True,
            "created_step": 0,
            "deadline_step": 8,
            "assigned_resource_id": None,
            "classified": False,
            "urgency_assessed": False,
            "verified": False,
            "verification_confidence": None,
            "reporter_type": "citizen",
            "reported_people_count": 5,
            "language_noise": False,
        }
    ],
    "active_assignments": [],
    "available_resources": [
        {"id": "RES-001", "type": "ambulance", "status": "available", "can_traverse_flood": False},
        {"id": "RES-002", "type": "rescue_boat", "status": "available", "can_traverse_flood": True},
    ],
    "zones": [
        {"id": "ZONE-A", "name": "Market District", "severity": 2, "access_blocked": False,
         "flood_depth_level": 2, "comms_blackout": False, "open_incidents": 1},
    ],
    "recent_changes": [],
    "warnings": [],
    "available_tools": ["call_intake_agent", "send_resource", "get_resources"],
    "last_action_result": None,
    "last_action_error": None,
    "weather_severity": 2,
    "situation_brief_submitted": False,
}

action = _heuristic_action(obs)
assert action["tool"] == "call_intake_agent", f"Expected intake, got {action['tool']}"
assert action["args"]["report_id"] == "RPT-001"
print(f"  Action: {action}")
print("  PASS")

# ============================================================
# TEST 2: Heuristic — classified report, flood-aware dispatch
# ============================================================
print()
print("=" * 60)
print("TEST 2: Heuristic — flood-aware dispatch")
print("=" * 60)

obs2 = dict(obs)
obs2["pending_reports"] = [dict(obs["pending_reports"][0])]
obs2["pending_reports"][0]["classified"] = True
obs2["pending_reports"][0]["urgency_assessed"] = True
obs2["pending_reports"][0]["verified"] = True
obs2["pending_reports"][0]["verification_confidence"] = 0.85
obs2["pending_reports"][0]["category"] = "flood"
obs2["pending_reports"][0]["status"] = "triaged"

action = _heuristic_action(obs2)
assert action["tool"] == "send_resource"
# Should pick the flood-capable resource (RES-002) since zone has flood_depth=2
assert action["args"]["resource_id"] == "RES-002", f"Expected RES-002 (boat), got {action['args']['resource_id']}"
print(f"  Action: {action}")
print("  PASS — picked flood-capable resource for flooded zone")

# ============================================================
# TEST 3: Heuristic — flag low-confidence report
# ============================================================
print()
print("=" * 60)
print("TEST 3: Heuristic — flag low-confidence report")
print("=" * 60)

obs3 = dict(obs)
obs3["pending_reports"] = [dict(obs["pending_reports"][0])]
obs3["pending_reports"][0]["classified"] = True
obs3["pending_reports"][0]["verified"] = True
obs3["pending_reports"][0]["verification_confidence"] = 0.15

action = _heuristic_action(obs3)
assert action["tool"] == "mark_false_report"
assert action["args"]["report_id"] == "RPT-001"
print(f"  Action: {action}")
print("  PASS")

# ============================================================
# TEST 4: Observation summary
# ============================================================
print()
print("=" * 60)
print("TEST 4: Observation summary for LLM")
print("=" * 60)

summary = _summarize_observation(obs)
assert "ZONE-A" in summary
assert "RPT-001" in summary
assert "FLOODED" in summary
assert "flood-capable" in summary
print(f"  Summary length: {len(summary)} chars")
print(f"  First 200 chars: {summary[:200]}...")
print("  PASS")

# ============================================================
# TEST 5: Full heuristic episode run
# ============================================================
print()
print("=" * 60)
print("TEST 5: Full heuristic episode (all 3 tasks)")
print("=" * 60)

for task in ["task1_flood_easy", "task2_storm_medium", "task3_cascade_hard"]:
    # Capture stdout (log output)
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    result = run_task(task, seed=42, use_llm=False)

    sys.stdout = old_stdout
    log_output = captured.getvalue()

    # Verify log format
    assert "[START]" in log_output, f"Missing [START] log for {task}"
    assert "[STEP]" in log_output, f"Missing [STEP] log for {task}"
    assert "[END]" in log_output, f"Missing [END] log for {task}"

    # Verify result
    assert "score" in result
    assert "total_reward" in result
    assert "critical_missed" in result
    assert "reports_resolved" in result
    assert result["steps"] > 0

    # Parse START log — key=value format
    start_line = [l for l in log_output.split("\n") if l.startswith("[START]")][0]
    assert f"task={task}" in start_line, f"[START] missing task name"
    assert "env=" in start_line, f"[START] missing env field"
    assert "model=" in start_line, f"[START] missing model field"

    # Parse END log — key=value format
    end_line = [l for l in log_output.split("\n") if l.startswith("[END]")][0]
    assert "success=" in end_line, f"[END] missing success field"
    assert "steps=" in end_line, f"[END] missing steps field"
    assert "score=" in end_line, f"[END] missing score field"
    assert "rewards=" in end_line, f"[END] missing rewards field"

    print(f"  {task}: score={result['score']:.4f}  "
          f"reward={result['total_reward']:.1f}  "
          f"resolved={result['reports_resolved']}/{result['total_reports']}  "
          f"steps={result['steps']}  "
          f"logs_ok=✓")

print("  PASS — all tasks run with correct log format")

print()
print("=== ALL PHASE 8 TESTS PASSED ===")
