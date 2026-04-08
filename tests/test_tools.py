import pytest

from src.env.scenarios import generate_scenario
from src.env.state import WorldState
from src.env.tool_registry import execute_tool


def _advance_to_step(state: WorldState, target_step: int) -> None:
    for step in range(target_step + 1):
        state.current_step = step
        state.advance_time()


def test_registry_handles_unknown_tool_and_missing_args() -> None:
    state = WorldState.from_scenario(generate_scenario("task1_flood_easy", seed=42))

    result, reward = execute_tool(state, "does_not_exist", {})
    assert "Unknown tool" in result
    assert reward == -0.5

    result, reward = execute_tool(state, "send_resource", {"resource_id": "RES-001"})
    assert "missing required arguments" in result
    assert reward == -0.5


def test_call_intake_agent_sets_report_intake_flags() -> None:
    state = WorldState.from_scenario(generate_scenario("task2_storm_medium", seed=42))
    _advance_to_step(state, 2)

    pending = state.get_pending_reports()
    assert pending, "Expected at least one pending report"
    report = pending[0]

    result, reward = execute_tool(state, "call_intake_agent", {"report_id": report.id})

    assert "Intake Agent Report" in result
    assert reward > 0.0

    updated = state.get_report(report.id)
    assert updated.classified is True
    assert updated.urgency_assessed is True
    assert updated.verified is True


def test_send_resource_with_matching_type_rewards_success() -> None:
    state = WorldState.from_scenario(generate_scenario("task2_storm_medium", seed=42))
    _advance_to_step(state, 5)

    real_pending = [
        r for r in state.get_pending_reports()
        if state.ground_truth.get(r.id, {}).get("verdict") == "real"
    ]
    assert real_pending, "Expected a real pending report"

    selected_report = None
    selected_resource = None
    for report in real_pending:
        required = state.ground_truth[report.id].get("required_resource")
        zone = state.get_zone(report.zone_id)
        for resource in state.get_available_resources():
            if resource.type.value != required:
                continue
            if zone.flood_depth_level >= 2 and not resource.can_traverse_flood:
                continue
            selected_report = report
            selected_resource = resource
            break
        if selected_report is not None:
            break

    if selected_report is None:
        pytest.skip("No compatible resource-report pair available for this seed")

    result, reward = execute_tool(
        state,
        "send_resource",
        {"resource_id": selected_resource.id, "report_id": selected_report.id},
    )

    assert "Assignment" in result
    assert reward >= 1.0
    assert state.get_report(selected_report.id).assigned_resource_id == selected_resource.id


def test_send_resource_rejects_non_flood_capable_resource_in_deep_flood() -> None:
    state = WorldState.from_scenario(generate_scenario("task1_flood_easy", seed=42))
    _advance_to_step(state, 0)

    report = state.get_pending_reports()[0]
    zone = state.get_zone(report.zone_id)
    zone.flood_depth_level = 2

    non_flood_resource = next((r for r in state.get_available_resources() if not r.can_traverse_flood), None)
    assert non_flood_resource is not None

    result, reward = execute_tool(
        state,
        "send_resource",
        {"resource_id": non_flood_resource.id, "report_id": report.id},
    )

    assert "cannot traverse flood" in result
    assert reward == -0.5


def test_mark_false_report_gives_positive_reward_for_false_or_duplicate() -> None:
    state = WorldState.from_scenario(generate_scenario("task2_storm_medium", seed=42))
    _advance_to_step(state, state.max_steps - 1)

    candidates = [
        r for r in state.get_pending_reports()
        if state.ground_truth.get(r.id, {}).get("verdict") in ("false", "duplicate")
    ]
    if not candidates:
        pytest.skip("No visible false/duplicate pending reports for this seed")

    target = candidates[0]
    result, reward = execute_tool(
        state,
        "mark_false_report",
        {"report_id": target.id, "reason": "low confidence and inconsistent details"},
    )

    assert "correct" in result
    assert reward == 1.0


def test_close_case_prematurely_penalizes_real_report() -> None:
    state = WorldState.from_scenario(generate_scenario("task1_flood_easy", seed=42))
    _advance_to_step(state, 0)

    real_pending = [
        r for r in state.get_pending_reports()
        if state.ground_truth.get(r.id, {}).get("verdict") == "real"
    ]
    assert real_pending, "Expected at least one real pending report"

    target = real_pending[0]
    result, reward = execute_tool(
        state,
        "close_case",
        {"report_id": target.id, "resolution_note": "closing early for test"},
    )

    assert "without dispatch" in result
    assert reward == -0.5


def test_reroute_resource_updates_blocked_assignment() -> None:
    state = WorldState.from_scenario(generate_scenario("task1_flood_easy", seed=42))

    report = state.reports["RPT-001"]
    resource = state.resources["RES-001"]
    state.current_step = 0
    assignment = state.create_assignment(resource.id, report.id, travel_steps=2)
    assignment.route_status = "blocked"
    assignment.stuck = True

    result, reward = execute_tool(
        state,
        "reroute_resource",
        {"resource_id": resource.id, "route_hint": "use upland bypass"},
    )

    assert "rerouted" in result
    assert reward == 0.5
    assert state.assignments[assignment.id].route_status.value == "rerouted"
    assert state.assignments[assignment.id].stuck is False
