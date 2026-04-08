from pathlib import Path

import yaml

from src.env.environment import DisasterReliefEnv
from src.env.scenarios import SUPPORTED_TASKS


ROOT = Path(__file__).resolve().parents[1]


def test_masterplan_openenv_core_contract() -> None:
    env = DisasterReliefEnv()

    # Round-1 core contract: typed env + reset/step/state/grade style API.
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "get_state")
    assert hasattr(env, "grade")

    tasks = env.supported_tasks()
    assert len(tasks) >= 3
    assert tasks == ["task1_flood_easy", "task2_storm_medium", "task3_cascade_hard"]


def test_masterplan_grade_range_for_all_tasks() -> None:
    env = DisasterReliefEnv()
    for task in SUPPORTED_TASKS:
        env.reset(task, seed=42)
        while not env.is_done:
            env.step({"tool": "get_resources", "args": {}})
        score = env.grade()["score"]
        assert 0.0 <= score <= 1.0


def test_masterplan_openenv_yaml_consistency() -> None:
    spec = yaml.safe_load((ROOT / "openenv.yaml").read_text(encoding="utf-8"))

    assert spec["name"] == "disaster-relief-coordination"
    assert "observation" in spec
    assert "action" in spec
    assert "reward" in spec
    assert "tasks" in spec
    assert "endpoints" in spec

    task_names = [task["name"] for task in spec["tasks"]]
    assert task_names == ["task1_flood_easy", "task2_storm_medium", "task3_cascade_hard"]


def test_masterplan_hf_space_runtime_artifacts_present() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "EXPOSE 7860" in dockerfile
    assert "uvicorn" in dockerfile
    assert "app_port: 7860" in readme
    assert "sdk: docker" in readme


def test_masterplan_inference_env_var_contract_and_tags() -> None:
    src = (ROOT / "inference.py").read_text(encoding="utf-8")

    # Required env vars + OpenAI client usage in baseline script.
    assert 'os.environ.get("API_BASE_URL"' in src
    assert 'os.environ.get("MODEL_NAME"' in src
    assert 'os.environ.get("HF_TOKEN"' in src
    assert "from openai import OpenAI" in src

    # Required logging tags appear.
    assert "[START]" in src
    assert "[STEP]" in src
    assert "[END]" in src


def test_masterplan_inference_offline_runs_all_tasks() -> None:
    from inference import run_task

    for task in SUPPORTED_TASKS:
        result = run_task(task_name=task, seed=42, use_llm=False)
        assert result["task"] == task
        assert result["steps"] > 0
        assert 0.0 <= result["score"] <= 1.0
