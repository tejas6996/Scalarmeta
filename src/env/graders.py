"""
Disaster Relief Coordination — Episode Graders
================================================
Each task has a grader that produces a final score in [0.0, 1.0]
from the episode's WorldState after all steps are complete.

Same seed + same actions = same score (deterministic).
A do-nothing agent scores ~0.0, a reasonable agent >0.5.
"""

from __future__ import annotations

from typing import Dict

from src.env.models import AssignmentStatus, ReportStatus, ReportVerdict
from src.env.state import WorldState


# ---------------------------------------------------------------------------
# Base grader
# ---------------------------------------------------------------------------

class BaseGrader:
    """Base class with shared scoring sub-routines."""

    def grade(self, state: WorldState) -> float:
        raise NotImplementedError

    # --- Sub-score helpers ---

    def _resolution_score(self, state: WorldState) -> float:
        """Fraction of real reports that were resolved."""
        real_ids = [
            rid for rid, gt in state.ground_truth.items()
            if gt.get("verdict") == "real"
        ]
        if not real_ids:
            return 1.0
        resolved = sum(
            1 for rid in real_ids
            if state.reports[rid].status == ReportStatus.RESOLVED
        )
        return resolved / len(real_ids)

    def _critical_score(self, state: WorldState) -> float:
        """Fraction of critical reports resolved before deadline."""
        critical_ids = [
            rid for rid, gt in state.ground_truth.items()
            if gt.get("is_critical", False) and gt.get("verdict") == "real"
        ]
        if not critical_ids:
            return 1.0

        handled = 0
        for rid in critical_ids:
            rpt = state.reports[rid]
            if rpt.status == ReportStatus.RESOLVED:
                # Bonus if resolved before deadline
                if rpt.deadline_step is None or (
                    rpt.resolved_step is not None and rpt.resolved_step <= rpt.deadline_step
                ):
                    handled += 1
                else:
                    handled += 0.5  # resolved but late
        return handled / len(critical_ids)

    def _efficiency_score(self, state: WorldState) -> float:
        """
        Efficiency = how well resources were used.
        Penalizes: wasted dispatches to false reports, failed assignments, idle resources.
        """
        total_assignments = len(state.assignments)
        if total_assignments == 0:
            return 0.0  # no action taken = no efficiency

        successful = sum(
            1 for a in state.assignments.values()
            if a.status == AssignmentStatus.COMPLETED
        )
        failed = sum(
            1 for a in state.assignments.values()
            if a.status == AssignmentStatus.FAILED
        )

        # Penalize dispatches to false/duplicate reports
        wasted = 0
        for a in state.assignments.values():
            gt = state.ground_truth.get(a.report_id, {})
            if gt.get("verdict") in ("false", "duplicate"):
                wasted += 1

        efficiency = (successful - 0.5 * wasted - 0.3 * failed) / max(total_assignments, 1)
        return max(0.0, min(1.0, efficiency))

    def _verification_accuracy(self, state: WorldState) -> float:
        """
        How accurately the agent identified false/duplicate reports.
        Score = (correct_flags + correct_non_flags) / total_reports.
        """
        total = len(state.reports)
        if total == 0:
            return 1.0

        correct = 0
        for rid, rpt in state.reports.items():
            gt = state.ground_truth.get(rid, {})
            verdict = gt.get("verdict", "real")

            if verdict in ("false", "duplicate"):
                # Should have been flagged
                if rpt.status == ReportStatus.FALSE:
                    correct += 1  # correctly flagged
                # Not flagged but not dispatched either — neutral (0.5 credit)
                elif rpt.status in (ReportStatus.PENDING, ReportStatus.TRIAGED):
                    correct += 0.3
            else:
                # Real report — should NOT be flagged
                if rpt.status != ReportStatus.FALSE:
                    correct += 1  # correctly not flagged
                # Flagged a real report — 0 credit

        return correct / total

    def _resource_correctness(self, state: WorldState) -> float:
        """
        Fraction of dispatches that used the correct resource type.
        """
        dispatches = [
            a for a in state.assignments.values()
            if a.status in (AssignmentStatus.COMPLETED, AssignmentStatus.EN_ROUTE,
                           AssignmentStatus.ON_SITE)
        ]
        if not dispatches:
            return 0.0

        correct = 0
        for a in dispatches:
            gt = state.ground_truth.get(a.report_id, {})
            required = gt.get("required_resource")
            if required is None:
                continue  # false report, skip
            resource = state.resources.get(a.resource_id)
            if resource and resource.type.value == required:
                correct += 1

        real_dispatches = sum(
            1 for a in dispatches
            if state.ground_truth.get(a.report_id, {}).get("verdict") == "real"
        )
        if real_dispatches == 0:
            return 0.0
        return correct / real_dispatches

    def _monitoring_score(self, state: WorldState) -> float:
        """
        How well the agent monitored ongoing operations.
        Based on: check_operation calls, stuck detection, timely closures.
        """
        # Count monitoring actions from memory
        monitor_actions = sum(
            1 for m in state.memory
            if m.tool_name in ("check_operation", "call_monitor_agent")
        )
        close_actions = sum(
            1 for m in state.memory
            if m.tool_name in ("close_case",)
        )

        # Desired: at least 1 monitor action per 3 active assignments made
        total_assignments = len(state.assignments)
        if total_assignments == 0:
            return 0.5  # no assignments = nothing to monitor

        desired_monitors = max(1, total_assignments // 3)
        monitor_ratio = min(1.0, monitor_actions / desired_monitors)

        # Desired: close_case called for completed assignments
        completed = sum(
            1 for a in state.assignments.values()
            if a.status == AssignmentStatus.COMPLETED
        )
        close_ratio = min(1.0, close_actions / max(completed, 1)) if completed > 0 else 0.5

        return 0.6 * monitor_ratio + 0.4 * close_ratio


# ---------------------------------------------------------------------------
# Task-specific graders
# ---------------------------------------------------------------------------

class Task1Grader(BaseGrader):
    """task1_flood_easy: 40% resolution + 30% critical + 30% efficiency."""

    def grade(self, state: WorldState) -> float:
        resolution = self._resolution_score(state)
        critical = self._critical_score(state)
        efficiency = self._efficiency_score(state)

        score = 0.40 * resolution + 0.30 * critical + 0.30 * efficiency
        return round(max(0.0, min(1.0, score)), 4)


class Task2Grader(BaseGrader):
    """task2_storm_medium: 30% resolution + 25% critical + 20% verification + 25% resource."""

    def grade(self, state: WorldState) -> float:
        resolution = self._resolution_score(state)
        critical = self._critical_score(state)
        verification = self._verification_accuracy(state)
        resource = self._resource_correctness(state)

        score = (
            0.30 * resolution
            + 0.25 * critical
            + 0.20 * verification
            + 0.25 * resource
        )
        return round(max(0.0, min(1.0, score)), 4)


class Task3Grader(BaseGrader):
    """task3_cascade_hard: 25% resolution + 25% critical + 20% verification + 15% resource + 15% monitoring."""

    def grade(self, state: WorldState) -> float:
        resolution = self._resolution_score(state)
        critical = self._critical_score(state)
        verification = self._verification_accuracy(state)
        resource = self._resource_correctness(state)
        monitoring = self._monitoring_score(state)

        score = (
            0.25 * resolution
            + 0.25 * critical
            + 0.20 * verification
            + 0.15 * resource
            + 0.15 * monitoring
        )
        return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------

GRADERS: Dict[str, BaseGrader] = {
    "task1_flood_easy": Task1Grader(),
    "task2_storm_medium": Task2Grader(),
    "task3_cascade_hard": Task3Grader(),
}


def grade_episode(state: WorldState) -> float:
    """
    Grade the completed episode using the appropriate task grader.

    Returns a score in [0.0, 1.0].
    """
    grader = GRADERS.get(state.task_name)
    if grader is None:
        raise ValueError(f"No grader for task '{state.task_name}'")
    return grader.grade(state)
