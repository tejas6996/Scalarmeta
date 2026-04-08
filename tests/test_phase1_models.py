import pytest
from pydantic import ValidationError

from src.env.models import (
    Action,
    EpisodeMemoryEntry,
    Report,
    ReportStatus,
    ReportVerdict,
    Resource,
    ResourceType,
    Zone,
)


def test_report_defaults_and_required_fields() -> None:
    report = Report(
        id="RPT-001",
        raw_text="Flooding near Riverside, 5 people trapped.",
        zone_id="ZONE-A",
        ground_truth_verdict=ReportVerdict.REAL,
        created_step=0,
    )

    assert report.status == ReportStatus.PENDING
    assert report.category == "unknown"
    assert report.urgency == 0
    assert report.classified is False
    assert report.urgency_assessed is False
    assert report.verified is False


@pytest.mark.parametrize("bad_urgency", [-1, 11])
def test_report_urgency_bounds_are_enforced(bad_urgency: int) -> None:
    with pytest.raises(ValidationError):
        Report(
            id="RPT-002",
            raw_text="Medical emergency at clinic.",
            zone_id="ZONE-A",
            urgency=bad_urgency,
            ground_truth_verdict=ReportVerdict.REAL,
            created_step=0,
        )


def test_resource_capacity_validation() -> None:
    with pytest.raises(ValidationError):
        Resource(
            id="RES-001",
            type=ResourceType.AMBULANCE,
            capacity=0,
        )


def test_zone_flood_depth_validation() -> None:
    with pytest.raises(ValidationError):
        Zone(
            id="ZONE-A",
            name="Riverside",
            flood_depth_level=4,
        )


def test_action_and_memory_defaults() -> None:
    action = Action(tool="get_resources")
    memory_entry = EpisodeMemoryEntry(step=1, tool_name="classify_report")

    assert action.args == {}
    assert memory_entry.actor == "coordinator"
    assert memory_entry.input_args == {}
    assert memory_entry.result_status == "success"
    assert memory_entry.reward == 0.0