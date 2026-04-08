import pytest

from src.env.models import ReportVerdict
from src.env.scenarios import SUPPORTED_TASKS, TASK_CONFIGS, generate_scenario


def _report_signature(report):
    return (
        report.id,
        report.zone_id,
        report.category,
        report.raw_text,
        report.created_step,
        report.ground_truth_verdict.value,
        report.ground_truth_duplicate_of,
        report.follow_up_of,
        report.deadline_step,
        report.required_resource_type.value if report.required_resource_type else None,
    )


@pytest.mark.parametrize("task_name", SUPPORTED_TASKS)
def test_task_counts_match_config(task_name: str) -> None:
    scenario = generate_scenario(task_name, seed=42)
    cfg = TASK_CONFIGS[task_name]

    assert len(scenario.zones) == cfg.zone_count
    assert len(scenario.resources) == len(cfg.resource_pool)
    assert len(scenario.reports) == cfg.report_count + cfg.follow_up_count
    assert len(scenario.ground_truth) == len(scenario.reports)


def test_scenario_generation_is_deterministic() -> None:
    s1 = generate_scenario("task2_storm_medium", seed=123)
    s2 = generate_scenario("task2_storm_medium", seed=123)

    assert [_report_signature(r) for r in s1.reports] == [_report_signature(r) for r in s2.reports]
    assert [z.model_dump() for z in s1.zones] == [z.model_dump() for z in s2.zones]
    assert [r.model_dump() for r in s1.resources] == [r.model_dump() for r in s2.resources]


def test_duplicate_reports_reference_real_report() -> None:
    scenario = generate_scenario("task3_cascade_hard", seed=42)
    real_ids = {r.id for r in scenario.reports if r.ground_truth_verdict == ReportVerdict.REAL}
    duplicates = [r for r in scenario.reports if r.ground_truth_verdict == ReportVerdict.DUPLICATE]

    assert duplicates, "Expected duplicate reports in hard task"
    for dup in duplicates:
        assert dup.ground_truth_duplicate_of in real_ids


def test_follow_up_reports_are_linked() -> None:
    scenario = generate_scenario("task3_cascade_hard", seed=42)
    follow_ups = [r for r in scenario.reports if r.follow_up_of is not None]

    assert len(follow_ups) == TASK_CONFIGS["task3_cascade_hard"].follow_up_count
    report_ids = {r.id for r in scenario.reports}
    for follow_up in follow_ups:
        assert follow_up.follow_up_of in report_ids


def test_unknown_task_raises_value_error() -> None:
    with pytest.raises(ValueError):
        generate_scenario("task_does_not_exist", seed=42)