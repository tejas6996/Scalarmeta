"""
Disaster Relief Coordination — Dispatch Tools
===============================================
Pure functions for querying inventory and deploying resources.
Each returns (result_text, reward_delta).
"""

from __future__ import annotations

from typing import Tuple

from src.env.models import (
    AssignmentStatus,
    ReportStatus,
    ReportVerdict,
    ResourceStatus,
    ResourceType,
    RouteStatus,
)
from src.env.state import WorldState


def get_resources(state: WorldState) -> Tuple[str, float]:
    """
    List all resources and their current status.

    This is a read-only query — no state mutation, no reward.
    """
    lines = ["Resource Inventory:"]
    for res in state.resources.values():
        status = res.status.value
        extra = ""
        if res.status == ResourceStatus.DEPLOYED:
            extra = f" → assigned to {res.assigned_report_id}"
        elif res.status == ResourceStatus.RETURNING:
            extra = f" (back at step {res.eta_available_step})"
        fuel = ""
        if res.fuel_steps_remaining is not None:
            fuel = f" | fuel={res.fuel_steps_remaining}"
        flood = " | flood-capable" if res.can_traverse_flood else ""
        lines.append(
            f"  {res.id}: {res.type.value} [{status}]{extra}"
            f" | capacity={res.capacity}{fuel}{flood}"
        )

    available = state.get_available_resources()
    lines.append(f"\nAvailable: {len(available)} / {len(state.resources)} total")

    return "\n".join(lines), 0.0


def send_resource(
    state: WorldState,
    resource_id: str,
    report_id: str,
) -> Tuple[str, float]:
    """
    Dispatch a resource to a report. Creates an assignment.

    Validates:
    - Resource exists and is available
    - Report exists and is pending/triaged
    - No existing active assignment for this report
    - Flood zone accessibility
    """
    resource = state.get_resource(resource_id)
    if resource is None:
        return f"Error: Resource '{resource_id}' not found.", -0.5

    report = state.get_report(report_id)
    if report is None:
        return f"Error: Report '{report_id}' not found.", -0.5

    # Resource must be available
    if resource.status != ResourceStatus.AVAILABLE:
        return (
            f"Error: Resource {resource_id} is currently {resource.status.value}. "
            f"Choose an available resource.",
            -0.5,
        )

    # Report must be in a dispatchable state
    if report.status not in (ReportStatus.PENDING, ReportStatus.TRIAGED):
        return (
            f"Error: Report {report_id} is already {report.status.value} — cannot dispatch.",
            -0.5,
        )

    # Check for existing active assignment
    existing = state.get_assignment_for_report(report_id)
    if existing is not None:
        return (
            f"Error: Report {report_id} already has an active assignment ({existing.id}).",
            -0.5,
        )

    # Check flood zone accessibility
    zone = state.get_zone(report.zone_id)
    if zone and zone.flood_depth_level >= 2 and not resource.can_traverse_flood:
        return (
            f"Error: Zone {zone.id} has flood depth {zone.flood_depth_level} — "
            f"resource {resource_id} ({resource.type.value}) cannot traverse flood. "
            f"Use a flood-capable resource (boat or helicopter).",
            -0.5,
        )

    # Determine travel time
    travel_steps = 2
    if zone and zone.severity >= 4:
        travel_steps += 1  # severe conditions slow travel

    # Create the assignment
    asg = state.create_assignment(resource_id, report_id, travel_steps)

    # Compute reward with resource matching confidence decay
    gt = state.ground_truth.get(report_id, {})
    verdict = gt.get("verdict", "real")
    reward = 0.0

    # Flood depth penalty: harder conditions reduce reward
    flood_depth = zone.flood_depth_level if zone else 0
    _FLOOD_PENALTY = {0: 0.0, 1: -0.25, 2: -0.5, 3: -0.75}
    flood_adj = _FLOOD_PENALTY.get(flood_depth, 0.0)

    # Reporter credibility bonus
    _CRED_BONUS = {
        "field_officer": 0.3,
        "automated_sensor": 0.2,
        "citizen": 0.0,
    }
    cred_bonus = _CRED_BONUS.get(report.reporter_type.value, 0.0)

    if verdict in ("false", "duplicate"):
        # Dispatching to a false/duplicate report
        reward = -1.0
        result = (
            f"Dispatched {resource_id} ({resource.type.value}) → report {report_id}. "
            f"Assignment {asg.id} created, ETA step {asg.expected_completion_step}. "
            f"WARNING: This report may not require a response."
        )
    else:
        # Real report — check resource type match
        gt_resource = gt.get("required_resource", None)
        if gt_resource and resource.type.value == gt_resource:
            reward = 2.0 + flood_adj + cred_bonus  # correct type, adjusted for conditions
            result = (
                f"Dispatched {resource_id} ({resource.type.value}) → report {report_id}. "
                f"Assignment {asg.id} created, ETA step {asg.expected_completion_step}. "
                f"Resource type matches the incident requirements."
            )
        elif gt_resource and resource.type.value != gt_resource:
            reward = -1.0  # wrong resource type
            result = (
                f"Dispatched {resource_id} ({resource.type.value}) → report {report_id}. "
                f"Assignment {asg.id} created, ETA step {asg.expected_completion_step}. "
                f"Note: This resource type may not be ideal for this incident."
            )
        else:
            reward = 1.0 + flood_adj + cred_bonus  # dispatched to real, no specific type
            result = (
                f"Dispatched {resource_id} ({resource.type.value}) → report {report_id}. "
                f"Assignment {asg.id} created, ETA step {asg.expected_completion_step}."
            )

    if asg.route_status == "blocked":
        result += f" Route is BLOCKED — travel delayed (+3 steps)."

    return result, reward


def reroute_resource(
    state: WorldState,
    resource_id: str,
    route_hint: str = "",
) -> Tuple[str, float]:
    """
    Attempt to reroute a deployed resource that is stuck or on a blocked route.

    Finds the active assignment for this resource and updates its route status.
    """
    resource = state.get_resource(resource_id)
    if resource is None:
        return f"Error: Resource '{resource_id}' not found.", -0.5

    if resource.status != ResourceStatus.DEPLOYED:
        return (
            f"Resource {resource_id} is {resource.status.value} — "
            f"only deployed resources can be rerouted.",
            0.0,
        )

    # Find the active assignment
    active_asg = None
    for asg in state.assignments.values():
        if asg.resource_id == resource_id and asg.status in (
            AssignmentStatus.EN_ROUTE, AssignmentStatus.ON_SITE
        ):
            active_asg = asg
            break

    if active_asg is None:
        return f"No active assignment found for resource {resource_id}.", 0.0

    if active_asg.route_status == RouteStatus.CLEAR:
        return (
            f"Assignment {active_asg.id} route is already clear — no reroute needed.",
            0.0,
        )

    if active_asg.route_status == RouteStatus.REROUTED:
        return (
            f"Assignment {active_asg.id} has already been rerouted.",
            0.0,
        )

    # Reroute: change status, extend ETA by 1
    active_asg.route_status = RouteStatus.REROUTED
    active_asg.stuck = False
    active_asg.expected_completion_step = max(
        active_asg.expected_completion_step,
        state.current_step + 2,
    )

    reward = 0.5  # proactive rerouting is rewarded
    result = (
        f"Resource {resource_id} rerouted via alternate path. "
        f"Assignment {active_asg.id} updated — new ETA step {active_asg.expected_completion_step}."
    )
    if route_hint:
        result += f" Route hint noted: '{route_hint}'."

    return result, reward
