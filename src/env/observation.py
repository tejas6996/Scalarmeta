"""
Disaster Relief Coordination — Observation Builder
====================================================
Constructs the Observation payload the LLM coordinator sees each step.
Strips all hidden ground truth; only exposes what the agent should know.
"""

from __future__ import annotations

from typing import List, Optional

from src.env.models import (
    AssignmentStatus,
    AssignmentSummary,
    Observation,
    ReportStatus,
    ReportSummary,
    ResourceSummary,
    ZoneSummary,
)
from src.env.state import WorldState
from src.env.tool_registry import get_tool_signatures


def build_observation(
    state: WorldState,
    last_action_result: Optional[str] = None,
    last_action_error: Optional[str] = None,
) -> Observation:
    """
    Build the observation the LLM sees at the current step.

    Parameters
    ----------
    state : WorldState
        Current world state.
    last_action_result : str or None
        Feedback from the previous tool call.
    last_action_error : str or None
        Error message if the previous action was invalid.
    """
    # --- Pending reports (visible, not resolved/expired/false) ---
    visible = state.get_visible_reports()
    pending_reports: List[ReportSummary] = []
    for r in visible:
        if r.status in (ReportStatus.RESOLVED, ReportStatus.FALSE, ReportStatus.EXPIRED):
            continue
        pending_reports.append(ReportSummary(
            id=r.id,
            raw_text=r.raw_text,
            zone_id=r.zone_id,
            category=r.category,
            urgency=r.urgency,
            status=r.status,
            is_critical=r.is_critical,
            created_step=r.created_step,
            deadline_step=r.deadline_step,
            assigned_resource_id=r.assigned_resource_id,
            classified=r.classified,
            urgency_assessed=r.urgency_assessed,
            verified=r.verified,
            verification_confidence=r.verification_confidence,
            reporter_type=r.reporter_type,
            reported_people_count=r.reported_people_count,
            language_noise=r.language_noise,
        ))

    # Sort: critical first, then by urgency desc, then by deadline asc
    pending_reports.sort(key=lambda r: (
        not r.is_critical,
        -r.urgency,
        r.deadline_step if r.deadline_step is not None else 9999,
    ))

    # --- Active assignments ---
    active_assignments: List[AssignmentSummary] = []
    for a in state.get_active_assignments():
        active_assignments.append(AssignmentSummary(
            id=a.id,
            resource_id=a.resource_id,
            report_id=a.report_id,
            created_step=a.created_step,
            route_status=a.route_status,
            status=a.status,
            expected_completion_step=a.expected_completion_step,
            stuck=a.stuck,
        ))

    # --- All resources (agent needs full picture) ---
    resources: List[ResourceSummary] = []
    for res in state.resources.values():
        resources.append(ResourceSummary(
            id=res.id,
            type=res.type,
            status=res.status,
            assigned_report_id=res.assigned_report_id,
            location=res.location,
            can_traverse_flood=res.can_traverse_flood,
        ))

    # --- Zone summaries ---
    zones: List[ZoneSummary] = []
    for z in state.zones.values():
        zones.append(ZoneSummary(
            id=z.id,
            name=z.name,
            severity=z.severity,
            access_blocked=z.access_blocked,
            flood_depth_level=z.flood_depth_level,
            comms_blackout=z.comms_blackout,
            open_incidents=z.open_incidents,
        ))

    return Observation(
        step=state.current_step,
        max_steps=state.max_steps,
        task_name=state.task_name,
        pending_reports=pending_reports,
        active_assignments=active_assignments,
        available_resources=resources,
        zones=zones,
        recent_changes=list(state.recent_changes),
        warnings=list(state.warnings),
        available_tools=get_tool_signatures(),
        last_action_result=last_action_result,
        last_action_error=last_action_error,
        weather_severity=state.weather_severity,
        situation_brief_submitted=state.situation_brief_submitted,
    )
