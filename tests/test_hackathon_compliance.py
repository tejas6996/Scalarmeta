import re
import json

from fastapi.testclient import TestClient

import inference
from app import app
from src.env.environment import DisasterReliefEnv


EXPECTED_DISASTER_TASKS = [
    "task1_flood_easy",
    "task2_storm_medium",
    "task3_cascade_hard",
]


def _run_fixed_policy(task_name: str):
    env = DisasterReliefEnv()
    obs = env.reset(task_name)
    done = False
    rewards = []

    while not done:
        result = env.step({"tool": "get_resources", "args": {}})
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        rewards.append(round(reward, 6))

    grade = env.grade()
    return rewards, env.get_state(), grade["score"]


def test_api_exposes_required_endpoints_smoke() -> None:
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/tasks").status_code == 200

    reset = client.post("/reset", json={"task_name": DisasterReliefEnv.supported_tasks()[0]})
    assert reset.status_code == 200

    _obs = reset.json()
    step = client.post(
        "/step",
        json={"tool": "get_resources", "args": {}},
    )
    assert step.status_code == 200
    step_payload = step.json()
    assert set(step_payload.keys()) == {"observation", "reward", "done", "info"}

    assert client.get("/state").status_code == 200
    assert client.post("/grade").status_code == 200


def test_round1_tasks_match_project_docs() -> None:
    client = TestClient(app)
    tasks = client.get("/tasks").json()["tasks"]
    assert tasks == EXPECTED_DISASTER_TASKS


def test_reset_accepts_documented_disaster_task() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"task_name": "task1_flood_easy"})
    assert response.status_code == 200


def test_inference_log_format_matches_round1_requirement(capsys) -> None:
    inference.log_start("task1_flood_easy")
    inference.log_step(
        step=0,
        observation={"dummy": True},
        action={"tool": "classify_report", "args": {"report_id": "RPT-001"}},
        reward=0.5,
        done=False,
    )
    inference.log_end(
        task_name="task1_flood_easy",
        score=0.75,
        total_reward=5.5,
        critical_missed=0,
    )

    lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 3

    assert lines[0].startswith("[START] ")
    assert lines[1].startswith("[STEP] ")
    assert lines[2].startswith("[END] ")

    start_payload = json.loads(lines[0].replace("[START] ", "", 1))
    step_payload = json.loads(lines[1].replace("[STEP] ", "", 1))
    end_payload = json.loads(lines[2].replace("[END] ", "", 1))

    assert start_payload["task"] == "task1_flood_easy"
    assert step_payload["step"] == 0
    assert isinstance(step_payload["done"], bool)
    assert end_payload["task"] == "task1_flood_easy"
    assert "score" in end_payload


def test_same_actions_produce_same_trajectory() -> None:
    for task_name in DisasterReliefEnv.supported_tasks():
        rewards_a, state_a, score_a = _run_fixed_policy(task_name)
        rewards_b, state_b, score_b = _run_fixed_policy(task_name)

        assert rewards_a == rewards_b
        assert state_a == state_b
        assert score_a == score_b