"""
Disaster Relief Coordination — Monitor Tools
==============================================
Pure functions for checking operations, closing cases, and flagging false reports.
Each returns (result_text, reward_delta).
"""

from __future__ import annotations

from typing import Tuple

from src.env.models import (
    AssignmentStatus,
    ReportStatus,
    ReportVerdict,
)
from src.env.state import WorldState


def check_operation(state: WorldState, target_id: str) -> Tuple[str, float]:
    """
    Check the status of a report or assignment.

    Read-only query — no state mutation, no reward.
    Accepts either a report ID (RPT-xxx) or assignment ID (ASG-xxx).
    """
    # Try report first
    report = state.get_report(target_id)
    if report is not None:
        lines = [
            f"Report {report.id}: status={report.status.value}",
            f"  Zone: {report.zone_id} | Category: {report.category} | Urgency: {report.urgency}/10",
            f"  Created: step {report.created_step}",
        ]
        if report.deadline_step is not None:
            remaining = report.deadline_step - state.current_step
            lines.append(f"  Deadline: step {report.deadline_step} ({remaining} step(s) left)")
        if report.assigned_resource_id:
            lines.append(f"  Assigned resource: {report.assigned_resource_id}")
        if report.is_critical:
            lines.append("  *** CRITICAL PRIORITY ***")

        asg = state.get_assignment_for_report(target_id)
        if asg:
            lines.append(
                f"  Active assignment: {asg.id} | {asg.status.value} | "
                f"route={asg.route_status} | ETA step {asg.expected_completion_step}"
            )
            if asg.stuck:
                lines.append("  *** ASSIGNMENT IS STUCK ***")

        lines.append(
            f"  Intake: classified={report.classified} | urgency_assessed={report.urgency_assessed} "
            f"| verified={report.verified}"
        )
        if report.verification_confidence is not None:
            lines.append(f"  Verification confidence: {report.verification_confidence:.0%}")

        return "\n".join(lines), 0.0

    # Try assignment
    asg = state.assignments.get(target_id)
    if asg is not None:
        lines = [
            f"Assignment {asg.id}: status={asg.status.value}",
            f"  Resource: {asg.resource_id} → Report: {asg.report_id}",
            f"  Created: step {asg.created_step} | Route: {asg.route_status}",
            f"  Expected completion: step {asg.expected_completion_step}",
        ]
        if asg.stuck:
            lines.append("  *** STUCK / DELAYED ***")
        return "\n".join(lines), 0.0

    return f"Error: No report or assignment found with ID '{target_id}'.", -0.5


def close_case(
    state: WorldState,
    report_id: str,
    resolution_note: str = "",
) -> Tuple[str, float]:
    """
    Close a case by marking the report as resolved.

    Validates that the report is in a closable state (DISPATCHED with
    completed assignment, or ON_SITE).
    """
    report = state.get_report(report_id)
    if report is None:
        return f"Error: Report '{report_id}' not found.", -0.5

    # Already resolved/closed
    if report.status in (ReportStatus.RESOLVED, ReportStatus.FALSE, ReportStatus.EXPIRED):
        return f"Report {report_id} is already {report.status.value} — cannot close again.", 0.0

    # Check if there's a completed or on-site assignment
    asg = state.get_assignment_for_report(report_id)
    has_completed_asg = any(
        a.report_id == report_id and a.status == AssignmentStatus.COMPLETED
        for a in state.assignments.values()
    )
    on_site = asg and asg.status == AssignmentStatus.ON_SITE

    if report.status == ReportStatus.DISPATCHED and (has_completed_asg or on_site):
        # Legitimate closure
        state.resolve_report(report_id)

        gt = state.ground_truth.get(report_id, {})
        verdict = gt.get("verdict", "real")

        if verdict == "real":
            reward = 1.5
            result = f"Case {report_id} closed successfully. Resolution: {resolution_note or 'no note'}."
        else:
            # Closed a false/duplicate via full dispatch — wasteful but not penalized heavily
            reward = 0.5
            result = (
                f"Case {report_id} closed. Note: This report was likely a "
                f"{'duplicate' if verdict == 'duplicate' else 'false alarm'} — "
                f"resources may have been wasted."
            )
        return result, reward

    elif report.status in (ReportStatus.PENDING, ReportStatus.TRIAGED):
        # Premature closure — no dispatch was done
        state.resolve_report(report_id)

        gt = state.ground_truth.get(report_id, {})
        verdict = gt.get("verdict", "real")

        if verdict == "real":
            reward = -0.5  # premature closure of real report
            result = (
                f"Case {report_id} closed without dispatch. "
                f"WARNING: If this was a real incident, it may go unresolved."
            )
        else:
            reward = 0.5  # closing a false report without wasting resources is okay
            result = f"Case {report_id} closed without dispatch. Report appears non-critical."

        return result, reward

    elif report.status == ReportStatus.DISPATCHED and not (has_completed_asg or on_site):
        # Resource still en-route
        return (
            f"Cannot close report {report_id} — resource is still en-route. "
            f"Wait for arrival or check assignment status.",
            0.0,
        )

    return f"Report {report_id} is in state {report.status.value} — cannot close.", 0.0


def mark_false_report(
    state: WorldState,
    report_id: str,
    reason: str = "",
) -> Tuple[str, float]:
    """
    Flag a report as false alarm or duplicate.

    Should be called after verify_report suggests low confidence.
    """
    report = state.get_report(report_id)
    if report is None:
        return f"Error: Report '{report_id}' not found.", -0.5

    if report.status in (ReportStatus.RESOLVED, ReportStatus.FALSE, ReportStatus.EXPIRED):
        return f"Report {report_id} is already {report.status.value} — cannot flag.", 0.0

    if report.status == ReportStatus.DISPATCHED:
        return (
            f"Report {report_id} has an active dispatch — "
            f"cancel the assignment first or wait for completion.",
            0.0,
        )

    # Flag it
    state.mark_report_false(report_id)

    # Check ground truth
    gt = state.ground_truth.get(report_id, {})
    verdict = gt.get("verdict", "real")

    if verdict in ("false", "duplicate"):
        reward = 1.0  # correct identification
        result = (
            f"Report {report_id} flagged as false/duplicate (correct). "
            f"Reason: {reason or 'not specified'}. No resources wasted."
        )
    else:
        reward = -1.5  # incorrectly dismissed a real report
        result = (
            f"Report {report_id} flagged as false/duplicate. "
            f"WARNING: This report may have been genuine — "
            f"affected people may not receive help."
        )
        if report.is_critical:
            reward = -2.0  # even harsher for dismissing critical
            result += " *** THIS WAS A CRITICAL REPORT ***"

    return result, reward
