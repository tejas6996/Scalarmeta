import json

from fastapi.testclient import TestClient

from app import app
from inference import _heuristic_action, run_task
from src.env.environment import DisasterReliefEnv
from src.env.graders import grade_episode
from src.env.rewards import compute_no_action_reward
from src.env.scenarios import generate_scenario
from src.env.state import WorldState
from src.env.tool_registry import execute_tool


def test_phase5_reward_no_action_is_zero_total() -> None:
    state = WorldState.from_scenario(generate_scenario("task1_flood_easy", seed=42))
    state.current_step = 0
    state.advance_time()

    reward = compute_no_action_reward(state)
    assert reward.base == 0.5
    assert reward.total == 0.0


def test_phase5_reasonable_agent_scores_higher_than_do_nothing_task1() -> None:
    s1 = generate_scenario("task1_flood_easy", seed=42)
    do_nothing = WorldState.from_scenario(s1)
    for step in range(do_nothing.max_steps):
        do_nothing.current_step = step
        do_nothing.advance_time()
    score_do_nothing = grade_episode(do_nothing)

    s2 = generate_scenario("task1_flood_easy", seed=42)
    active = WorldState.from_scenario(s2)
    for step in range(active.max_steps):
        active.current_step = step
        active.advance_time()
        for rpt in active.get_pending_reports():
            if not rpt.classified:
                execute_tool(active, "call_intake_agent", {"report_id": rpt.id})
        pending = active.get_pending_reports()
        available = active.get_available_resources()
        for rpt in sorted(pending, key=lambda r: (-r.urgency, r.id)):
            if not available:
                break
            if rpt.verified and rpt.verification_confidence is not None and rpt.verification_confidence < 0.3:
                execute_tool(active, "mark_false_report", {"report_id": rpt.id, "reason": "low confidence"})
                continue
            res = available.pop(0)
            execute_tool(active, "send_resource", {"resource_id": res.id, "report_id": rpt.id})
    score_active = grade_episode(active)

    assert score_active >= score_do_nothing


def test_phase6_environment_reset_step_grade_cycle() -> None:
    env = DisasterReliefEnv()
    obs = env.reset("task1_flood_easy", seed=42)

    assert obs["task_name"] == "task1_flood_easy"
    assert "pending_reports" in obs
    assert "available_resources" in obs

    result = env.step({"tool": "get_resources", "args": {}})
    assert set(result.keys()) == {"observation", "reward", "done", "info"}

    while not env.is_done:
        result = env.step({"tool": "get_resources", "args": {}})

    grade = env.grade()
    assert 0.0 <= grade["score"] <= 1.0


def test_phase7_fastapi_contract_smoke() -> None:
    client = TestClient(app)

    assert client.get("/").status_code == 200
    tasks_resp = client.get("/tasks")
    assert tasks_resp.status_code == 200
    tasks = tasks_resp.json()["tasks"]
    assert tasks == ["task1_flood_easy", "task2_storm_medium", "task3_cascade_hard"]

    reset = client.post("/reset", json={"task_name": tasks[0], "seed": 42})
    assert reset.status_code == 200

    step = client.post("/step", json={"tool": "get_resources", "args": {}})
    assert step.status_code == 200
    payload = step.json()
    assert set(payload.keys()) == {"observation", "reward", "done", "info"}

    assert client.get("/state").status_code == 200
    assert client.post("/grade").status_code == 200


def test_phase8_inference_heuristic_and_offline_run() -> None:
    obs = {
        "step": 0,
        "max_steps": 12,
        "task_name": "task1_flood_easy",
        "pending_reports": [
            {
                "id": "RPT-001",
                "raw_text": "Flooding near Riverside",
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
            {
                "id": "ZONE-A",
                "name": "Riverside",
                "severity": 2,
                "access_blocked": False,
                "flood_depth_level": 2,
                "comms_blackout": False,
                "open_incidents": 1,
            }
        ],
        "recent_changes": [],
        "warnings": [],
        "available_tools": [],
        "last_action_result": None,
        "last_action_error": None,
        "weather_severity": 2,
        "situation_brief_submitted": False,
    }

    action = _heuristic_action(obs)
    assert action["tool"] == "call_intake_agent"
    assert action["args"]["report_id"] == "RPT-001"

    summary = run_task("task1_flood_easy", seed=42, use_llm=False)
    assert summary["task"] == "task1_flood_easy"
    assert summary["steps"] > 0
    assert 0.0 <= summary["score"] <= 1.0
    assert isinstance(summary["reports_resolved"], int)

    json.dumps(summary)