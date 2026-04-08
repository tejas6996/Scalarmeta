from pathlib import Path

import inference
from src.env.environment import DisasterReliefEnv


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dockerfile_targets_hf_space_port_and_server_entrypoint() -> None:
    dockerfile = _read(ROOT / "Dockerfile")

    assert "EXPOSE 7860" in dockerfile
    assert "PORT=7860" in dockerfile
    assert 'CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]' in dockerfile


def test_openenv_yaml_contains_required_top_level_sections() -> None:
    spec = _read(ROOT / "openenv.yaml")

    required_keys = [
        "name:",
        "observation:",
        "action:",
        "reward:",
        "tasks:",
        "endpoints:",
    ]
    for key in required_keys:
        assert key in spec, f"Missing '{key}' in openenv.yaml"

    task_name_occurrences = spec.count("- name:")
    assert task_name_occurrences >= 3


def test_inference_reads_required_env_vars_and_uses_openai_client() -> None:
    source = _read(ROOT / "inference.py")

    assert 'os.environ.get("API_BASE_URL"' in source
    assert 'os.environ.get("MODEL_NAME"' in source
    assert 'os.environ.get("HF_TOKEN"' in source
    assert "from openai import OpenAI" in source
    assert "client = OpenAI(" in source


def test_inference_runs_in_heuristic_mode_without_token() -> None:
    summary = inference.run_task(task_name=DisasterReliefEnv.supported_tasks()[0], use_llm=False)

    assert summary["task"] in DisasterReliefEnv.supported_tasks()
    assert summary["steps"] > 0
    assert 0.0 <= summary["score"] <= 1.0


def test_validation_scripts_include_hf_docker_openenv_checks() -> None:
    sh_script = _read(ROOT / "validate-submission.sh")
    ps_script = _read(ROOT / "validate-submission.ps1")

    for script in (sh_script, ps_script):
        assert "/reset" in script
        assert "docker build" in script
        assert "openenv validate" in script