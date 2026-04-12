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
        False alarm detection quality using F1-score.
        Precision = correct flags / total flags (avoids flag-everything exploit).
        Recall = correct flags / total actual false+dup reports (avoids ignore-everything).
        Returns harmonic mean (F1). If no false reports exist, returns 1.0.
        """
        # Ground truth counts
        actual_false_ids = {
            rid for rid, gt in state.ground_truth.items()
            if gt.get("verdict") in ("false", "duplicate")
        }
        if not actual_false_ids:
            return 1.0  # nothing to detect

        # Agent's flags
        flagged_ids = {
            rid for rid, rpt in state.reports.items()
            if rpt.status == ReportStatus.FALSE
        }

        true_positives = len(flagged_ids & actual_false_ids)
        false_positives = len(flagged_ids - actual_false_ids)
        false_negatives = len(actual_false_ids - flagged_ids)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _counterfactual_penalty(self, state: WorldState) -> float:
        """
        For each expired critical report, check if a correct resource type
        was available at any step before the deadline. If so, the agent
        *could* have acted but didn't — apply a harsher penalty.

        Returns a score in [0.0, 1.0] where 1.0 = no preventable misses.
        """
        critical_ids = [
            rid for rid, gt in state.ground_truth.items()
            if gt.get("is_critical", False) and gt.get("verdict") == "real"
        ]
        if not critical_ids:
            return 1.0

        preventable = 0
        total_expired_critical = 0

        for rid in critical_ids:
            rpt = state.reports[rid]
            if rpt.status != ReportStatus.EXPIRED:
                continue
            total_expired_critical += 1

            gt = state.ground_truth.get(rid, {})
            required = gt.get("required_resource")
            if required is None:
                continue

            # Check availability log: was the correct resource type available
            # at any step between report creation and deadline?
            deadline = rpt.deadline_step or state.max_steps
            for snapshot in state.availability_log:
                snap_step = snapshot["step"]
                if snap_step < rpt.created_step:
                    continue
                if snap_step > deadline:
                    break
                if required in snapshot.get("available", {}):
                    preventable += 1
                    break

        if total_expired_critical == 0:
            return 1.0  # no critical expirations

        # Score: 1.0 if none were preventable, 0.0 if all were
        return 1.0 - (preventable / total_expired_critical)

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
    """task2_storm_medium: 30% resolution + 25% critical + 15% F1 + 15% resource + 15% counterfactual."""

    def grade(self, state: WorldState) -> float:
        resolution = self._resolution_score(state)
        critical = self._critical_score(state)
        verification = self._verification_accuracy(state)
        resource = self._resource_correctness(state)
        counterfactual = self._counterfactual_penalty(state)

        score = (
            0.30 * resolution
            + 0.25 * critical
            + 0.15 * verification
            + 0.15 * resource
            + 0.15 * counterfactual
        )
        return round(max(0.0, min(1.0, score)), 4)


class Task3Grader(BaseGrader):
    """task3_cascade_hard: 30% resolution + 25% critical + 15% F1 + 10% resource + 10% monitoring + 10% counterfactual."""

    def grade(self, state: WorldState) -> float:
        resolution = self._resolution_score(state)
        critical = self._critical_score(state)
        verification = self._verification_accuracy(state)
        resource = self._resource_correctness(state)
        monitoring = self._monitoring_score(state)
        counterfactual = self._counterfactual_penalty(state)

        score = (
            0.30 * resolution
            + 0.25 * critical
            + 0.15 * verification
            + 0.10 * resource
            + 0.10 * monitoring
            + 0.10 * counterfactual
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
