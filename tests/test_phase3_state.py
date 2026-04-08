import pytest

from src.env.models import AssignmentStatus, ReportStatus, ResourceStatus
from src.env.scenarios import generate_scenario
from src.env.state import WorldState


def _advance_to_step(state: WorldState, target_step: int) -> None:
    for step in range(target_step + 1):
        state.current_step = step
        state.advance_time()


def test_visibility_respects_comms_blackout_until_restore() -> None:
    scenario = generate_scenario("task2_storm_medium", seed=42)
    blackout_zones = [z for z in scenario.zones if z.comms_blackout and z.comms_restored_step is not None]
    if not blackout_zones:
        pytest.skip("No blackout zones generated for this seed")

    blackout_zone = blackout_zones[0]
    candidates = [r for r in scenario.reports if r.zone_id == blackout_zone.id]
    if not candidates:
        pytest.skip("No reports generated in blackout zone for this seed")

    report = min(candidates, key=lambda r: r.created_step)
    state = WorldState.from_scenario(scenario)

    before_restore = max(0, blackout_zone.comms_restored_step - 1)
    _advance_to_step(state, before_restore)
    assert report.id not in {r.id for r in state.get_visible_reports()}

    _advance_to_step(state, blackout_zone.comms_restored_step)
    if report.created_step <= blackout_zone.comms_restored_step:
        assert report.id in {r.id for r in state.get_visible_reports()}


def test_create_assignment_updates_report_and_resource() -> None:
    scenario = generate_scenario("task2_storm_medium", seed=42)
    state = WorldState.from_scenario(scenario)

    report = scenario.reports[0]
    resource = scenario.resources[0]
    zone = state.get_zone(report.zone_id)

    state.current_step = 0
    assignment = state.create_assignment(resource.id, report.id, travel_steps=2)

    assert assignment.status == AssignmentStatus.EN_ROUTE
    assert state.get_report(report.id).status == ReportStatus.DISPATCHED
    assert state.get_report(report.id).assigned_resource_id == resource.id
    assert state.get_resource(resource.id).status == ResourceStatus.DEPLOYED

    expected_steps = 5 if zone.access_blocked else 2
    assert assignment.expected_completion_step == state.current_step + expected_steps


def test_assignment_lifecycle_completes_and_resource_returns() -> None:
    scenario = generate_scenario("task1_flood_easy", seed=42)
    state = WorldState.from_scenario(scenario)

    report = scenario.reports[0]
    resource = scenario.resources[0]

    state.current_step = 0
    assignment = state.create_assignment(resource.id, report.id, travel_steps=1)

    state.current_step = 1
    state.advance_time()
    assert state.assignments[assignment.id].status == AssignmentStatus.ON_SITE

    state.current_step = 2
    state.advance_time()
    assert state.assignments[assignment.id].status == AssignmentStatus.COMPLETED
    assert state.get_report(report.id).status == ReportStatus.RESOLVED
    assert state.get_resource(resource.id).status == ResourceStatus.RETURNING

    state.current_step = 3
    state.advance_time()
    assert state.get_resource(resource.id).status == ResourceStatus.AVAILABLE


def test_critical_report_expires_and_updates_counters() -> None:
    scenario = generate_scenario("task1_flood_easy", seed=42)
    state = WorldState.from_scenario(scenario)

    critical_reports = [r for r in scenario.reports if r.is_critical and r.deadline_step is not None]
    assert critical_reports, "Expected at least one critical report with a deadline"
    report = critical_reports[0]

    _advance_to_step(state, report.deadline_step)

    tracked = state.get_report(report.id)
    assert tracked.status == ReportStatus.EXPIRED
    assert state.reports_expired >= 1
    assert state.critical_missed >= 1