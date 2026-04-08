"""
Disaster Relief Coordination — World State
============================================
Mutable state container for a single episode. All tools read from and write to
this object. ``advance_time()`` is called once per step to progress the
simulation clock.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.env.models import (
    Assignment,
    AssignmentStatus,
    EpisodeMemoryEntry,
    Report,
    ReportStatus,
    Resource,
    ResourceStatus,
    Zone,
)
from src.env.scenarios import Scenario, TaskConfig


class WorldState:
    """Holds all mutable episode state and provides query/mutation helpers."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, scenario: Scenario) -> None:
        cfg: TaskConfig = scenario.task_config

        self.task_name: str = cfg.name
        self.max_steps: int = cfg.max_steps
        self.current_step: int = 0
        self.done: bool = False

        # Weather (static for the episode)
        self.weather_severity: int = cfg.weather_severity

        # Domain objects keyed by id for O(1) lookup
        self.reports: Dict[str, Report] = {r.id: r for r in scenario.reports}
        self.resources: Dict[str, Resource] = {r.id: r for r in scenario.resources}
        self.zones: Dict[str, Zone] = {z.id: z for z in scenario.zones}
        self.assignments: Dict[str, Assignment] = {}

        # Ground truth (hidden from agent)
        self.ground_truth: Dict[str, Any] = dict(scenario.ground_truth)

        # Reward bookkeeping
        self.total_reward: float = 0.0
        self.step_rewards: List[float] = []

        # Episode memory (structured log of actions & outcomes)
        self.memory: List[EpisodeMemoryEntry] = []

        # Recent-change buffer populated by advance_time(), consumed by obs builder
        self.recent_changes: List[str] = []

        # Warning buffer populated by advance_time()
        self.warnings: List[str] = []

        # Counters used by graders / state endpoint
        self.reports_resolved: int = 0
        self.reports_expired: int = 0
        self.reports_false_flagged: int = 0
        self.critical_missed: int = 0

        # Track previous actions for repeat-detection penalty
        self.last_action_key: Optional[str] = None

        # Situation brief flag
        self.situation_brief_submitted: bool = False

        # Internal assignment counter
        self._next_assignment_idx: int = 1

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "WorldState":
        """Create a fresh WorldState from a generated scenario."""
        return cls(scenario)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_visible_reports(self) -> List[Report]:
        """Reports whose created_step <= current_step and whose zone is
        not in comms blackout (unless comms have been restored)."""
        visible: List[Report] = []
        for r in self.reports.values():
            if r.created_step > self.current_step:
                continue
            zone = self.zones.get(r.zone_id)
            if zone and zone.comms_blackout:
                # Zone still blacked out — report not yet visible
                continue
            visible.append(r)
        return visible

    def get_pending_reports(self) -> List[Report]:
        """Visible reports that still need attention (PENDING or TRIAGED)."""
        return [
            r for r in self.get_visible_reports()
            if r.status in (ReportStatus.PENDING, ReportStatus.TRIAGED)
        ]

    def get_dispatched_reports(self) -> List[Report]:
        """Visible reports that are currently dispatched."""
        return [
            r for r in self.get_visible_reports()
            if r.status == ReportStatus.DISPATCHED
        ]

    def get_available_resources(self) -> List[Resource]:
        """Resources currently available for dispatch."""
        return [
            r for r in self.resources.values()
            if r.status == ResourceStatus.AVAILABLE
        ]

    def get_active_assignments(self) -> List[Assignment]:
        """Assignments that are still in progress (EN_ROUTE or ON_SITE)."""
        return [
            a for a in self.assignments.values()
            if a.status in (AssignmentStatus.EN_ROUTE, AssignmentStatus.ON_SITE)
        ]

    def get_report(self, report_id: str) -> Optional[Report]:
        return self.reports.get(report_id)

    def get_resource(self, resource_id: str) -> Optional[Resource]:
        return self.resources.get(resource_id)

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        return self.zones.get(zone_id)

    def get_assignment_for_report(self, report_id: str) -> Optional[Assignment]:
        for a in self.assignments.values():
            if a.report_id == report_id and a.status in (
                AssignmentStatus.EN_ROUTE, AssignmentStatus.ON_SITE
            ):
                return a
        return None

    # ------------------------------------------------------------------
    # Mutations (called by tools)
    # ------------------------------------------------------------------

    def create_assignment(
        self,
        resource_id: str,
        report_id: str,
        travel_steps: int = 2,
    ) -> Assignment:
        """Create a new assignment, updating the resource and report."""
        asg_id = f"ASG-{self._next_assignment_idx:03d}"
        self._next_assignment_idx += 1

        resource = self.resources[resource_id]
        report = self.reports[report_id]
        zone = self.zones[report.zone_id]

        # Determine route status based on zone conditions
        route_status = "clear"
        if zone.access_blocked:
            route_status = "blocked"
            travel_steps += 3  # penalty for blocked route

        asg = Assignment(
            id=asg_id,
            resource_id=resource_id,
            report_id=report_id,
            created_step=self.current_step,
            route_status=route_status,
            status=AssignmentStatus.EN_ROUTE,
            expected_completion_step=self.current_step + travel_steps,
        )

        self.assignments[asg_id] = asg

        # Update resource
        resource.status = ResourceStatus.DEPLOYED
        resource.assigned_report_id = report_id
        resource.location = report.zone_id

        # Update report
        report.status = ReportStatus.DISPATCHED
        report.assigned_resource_id = resource_id

        return asg

    def resolve_report(self, report_id: str) -> None:
        """Mark a report as resolved."""
        report = self.reports[report_id]
        report.status = ReportStatus.RESOLVED
        report.resolved_step = self.current_step
        self.reports_resolved += 1

        zone = self.zones.get(report.zone_id)
        if zone and zone.open_incidents > 0:
            zone.open_incidents -= 1

    def mark_report_false(self, report_id: str) -> None:
        """Mark a report as false alarm / duplicate."""
        report = self.reports[report_id]
        report.status = ReportStatus.FALSE
        report.resolved_step = self.current_step
        self.reports_false_flagged += 1

    def release_resource(self, resource_id: str) -> None:
        """Return a resource to available status at base."""
        resource = self.resources[resource_id]
        resource.status = ResourceStatus.AVAILABLE
        resource.assigned_report_id = None
        resource.location = "base"
        resource.eta_available_step = None

    def add_memory(self, entry: EpisodeMemoryEntry) -> None:
        """Append an entry to the episode memory log."""
        self.memory.append(entry)

    # ------------------------------------------------------------------
    # Time Advancement (called once per step)
    # ------------------------------------------------------------------

    def advance_time(self) -> None:
        """
        Progress the world by one step. This is the core simulation tick.

        Order of operations:
        1. Clear previous change/warning buffers
        2. Restore comms in zones whose blackout ends this step
        3. Clear blockages in zones whose blockage ends this step
        4. Surface newly visible reports
        5. Progress active assignments (arrive, complete)
        6. Handle fuel consumption for deployed resources
        7. Expire reports past their deadline
        8. Recompute zone incident counts
        9. Generate warnings for approaching deadlines & stuck ops
        """
        self.recent_changes.clear()
        self.warnings.clear()

        step = self.current_step

        # --- 1. Restore comms ---
        self._restore_comms(step)

        # --- 2. Clear blockages ---
        self._clear_blockages(step)

        # --- 3. Surface new reports ---
        self._surface_new_reports(step)

        # --- 4. Progress assignments ---
        self._progress_assignments(step)

        # --- 5. Fuel consumption ---
        self._consume_fuel(step)

        # --- 6. Expire overdue reports ---
        self._expire_overdue_reports(step)

        # --- 7. Recompute zone incident counts ---
        self._recompute_zone_incidents()

        # --- 8. Generate warnings ---
        self._generate_warnings(step)

        # --- 9. Check episode termination ---
        if step >= self.max_steps - 1:
            self.done = True

    # ------------------------------------------------------------------
    # advance_time sub-routines
    # ------------------------------------------------------------------

    def _restore_comms(self, step: int) -> None:
        for zone in self.zones.values():
            if zone.comms_blackout and zone.comms_restored_step is not None and step >= zone.comms_restored_step:
                zone.comms_blackout = False
                self.recent_changes.append(
                    f"Communications restored in {zone.name} ({zone.id})."
                )

    def _clear_blockages(self, step: int) -> None:
        for zone in self.zones.values():
            if zone.access_blocked and zone.blockage_clears_step is not None and step >= zone.blockage_clears_step:
                zone.access_blocked = False
                self.recent_changes.append(
                    f"Road blockage cleared in {zone.name} ({zone.id})."
                )
                # Un-block any assignments that were stuck due to this blockage
                for asg in self.assignments.values():
                    if asg.status == AssignmentStatus.EN_ROUTE and asg.route_status == "blocked":
                        rpt = self.reports.get(asg.report_id)
                        if rpt and rpt.zone_id == zone.id:
                            asg.route_status = "rerouted"
                            asg.stuck = False
                            # Extend ETA by 1 step for reroute overhead
                            asg.expected_completion_step = max(
                                asg.expected_completion_step, step + 1
                            )
                            self.recent_changes.append(
                                f"Assignment {asg.id} rerouted after blockage cleared."
                            )

    def _surface_new_reports(self, step: int) -> None:
        for report in self.reports.values():
            if report.created_step == step and report.status == ReportStatus.PENDING:
                zone = self.zones.get(report.zone_id)
                # Only surface if zone comms are up
                if zone and not zone.comms_blackout:
                    zone.last_contact_step = step
                    zone.open_incidents += 1
                    label = "CRITICAL " if report.is_critical else ""
                    self.recent_changes.append(
                        f"New {label}report {report.id} received from {zone.name}."
                    )

    def _progress_assignments(self, step: int) -> None:
        for asg in list(self.assignments.values()):
            if asg.status not in (AssignmentStatus.EN_ROUTE, AssignmentStatus.ON_SITE):
                continue

            # Check if route is blocked — assignment stalls
            if asg.route_status == "blocked":
                rpt = self.reports.get(asg.report_id)
                if rpt:
                    zone = self.zones.get(rpt.zone_id)
                    if zone and zone.access_blocked:
                        asg.stuck = True
                        continue

            # Arrival check
            if asg.status == AssignmentStatus.EN_ROUTE and step >= asg.expected_completion_step:
                asg.status = AssignmentStatus.ON_SITE
                self.recent_changes.append(
                    f"Resource arrived on site for assignment {asg.id} (report {asg.report_id})."
                )

            # On-site completion: 1 step on site then completed
            elif asg.status == AssignmentStatus.ON_SITE:
                asg.status = AssignmentStatus.COMPLETED
                self._complete_assignment(asg)

            # Stuck detection
            if asg.status == AssignmentStatus.EN_ROUTE and step > asg.expected_completion_step + 1:
                asg.stuck = True

    def _complete_assignment(self, asg: Assignment) -> None:
        """Handle an assignment reaching COMPLETED status."""
        report = self.reports.get(asg.report_id)
        resource = self.resources.get(asg.resource_id)

        if report and report.status == ReportStatus.DISPATCHED:
            report.status = ReportStatus.RESOLVED
            report.resolved_step = self.current_step
            self.reports_resolved += 1

        if resource:
            resource.status = ResourceStatus.RETURNING
            resource.eta_available_step = self.current_step + 1

        self.recent_changes.append(
            f"Assignment {asg.id} completed. Report {asg.report_id} resolved."
        )

    def _consume_fuel(self, step: int) -> None:
        for resource in self.resources.values():
            if resource.status == ResourceStatus.DEPLOYED and resource.fuel_steps_remaining is not None:
                resource.fuel_steps_remaining -= 1
                if resource.fuel_steps_remaining <= 0:
                    # Force resource to return — cancel its assignment
                    resource.status = ResourceStatus.RETURNING
                    resource.eta_available_step = step + 2
                    self.recent_changes.append(
                        f"Resource {resource.id} ran out of fuel — forced to return to base."
                    )
                    # Mark associated assignment as failed
                    for asg in self.assignments.values():
                        if (
                            asg.resource_id == resource.id
                            and asg.status in (AssignmentStatus.EN_ROUTE, AssignmentStatus.ON_SITE)
                        ):
                            asg.status = AssignmentStatus.FAILED
                            self.recent_changes.append(
                                f"Assignment {asg.id} failed — resource out of fuel."
                            )
                            # Revert report to triaged so it can be re-dispatched
                            rpt = self.reports.get(asg.report_id)
                            if rpt and rpt.status == ReportStatus.DISPATCHED:
                                rpt.status = ReportStatus.TRIAGED
                                rpt.assigned_resource_id = None

        # Return resources whose ETA has arrived
        for resource in self.resources.values():
            if (
                resource.status == ResourceStatus.RETURNING
                and resource.eta_available_step is not None
                and step >= resource.eta_available_step
            ):
                self.release_resource(resource.id)
                self.recent_changes.append(
                    f"Resource {resource.id} returned to base and is now available."
                )

    def _expire_overdue_reports(self, step: int) -> None:
        for report in self.reports.values():
            if (
                report.deadline_step is not None
                and step >= report.deadline_step
                and report.status in (ReportStatus.PENDING, ReportStatus.TRIAGED)
            ):
                report.status = ReportStatus.EXPIRED
                self.reports_expired += 1
                if report.is_critical:
                    self.critical_missed += 1
                self.recent_changes.append(
                    f"Report {report.id} EXPIRED — deadline missed at step {report.deadline_step}."
                )

    def _recompute_zone_incidents(self) -> None:
        """Recount open incidents per zone from report statuses."""
        counts: Dict[str, int] = {z_id: 0 for z_id in self.zones}
        for r in self.reports.values():
            if r.status in (
                ReportStatus.PENDING, ReportStatus.TRIAGED, ReportStatus.DISPATCHED
            ):
                if r.created_step <= self.current_step:
                    counts[r.zone_id] = counts.get(r.zone_id, 0) + 1
        for z_id, zone in self.zones.items():
            zone.open_incidents = counts.get(z_id, 0)

    def _generate_warnings(self, step: int) -> None:
        # Approaching deadlines
        for report in self.reports.values():
            if (
                report.deadline_step is not None
                and report.status in (ReportStatus.PENDING, ReportStatus.TRIAGED, ReportStatus.DISPATCHED)
                and 0 < report.deadline_step - step <= 2
            ):
                self.warnings.append(
                    f"DEADLINE WARNING: Report {report.id} expires at step {report.deadline_step} "
                    f"(only {report.deadline_step - step} step(s) left)."
                )

        # Stuck assignments
        for asg in self.assignments.values():
            if asg.stuck and asg.status in (AssignmentStatus.EN_ROUTE,):
                self.warnings.append(
                    f"STUCK: Assignment {asg.id} (resource→report {asg.report_id}) is stuck/delayed."
                )

        # Empty inventory by type
        available = self.get_available_resources()
        if not available:
            self.warnings.append("NO RESOURCES AVAILABLE — all resources are deployed or returning.")

        # Low fuel warnings
        for res in self.resources.values():
            if (
                res.status == ResourceStatus.DEPLOYED
                and res.fuel_steps_remaining is not None
                and res.fuel_steps_remaining <= 2
            ):
                self.warnings.append(
                    f"LOW FUEL: Resource {res.id} has only {res.fuel_steps_remaining} fuel step(s) left."
                )

    # ------------------------------------------------------------------
    # Summary helpers (for graders / state endpoint)
    # ------------------------------------------------------------------

    @property
    def total_reports(self) -> int:
        return len(self.reports)

    @property
    def critical_total(self) -> int:
        return sum(1 for r in self.reports.values() if r.is_critical)

    @property
    def active_assignments_count(self) -> int:
        return len(self.get_active_assignments())

    @property
    def resources_available_count(self) -> int:
        return len(self.get_available_resources())

    @property
    def resources_deployed_count(self) -> int:
        return sum(1 for r in self.resources.values() if r.status == ResourceStatus.DEPLOYED)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.env.scenarios import generate_scenario, SUPPORTED_TASKS

    for task in SUPPORTED_TASKS:
        scenario = generate_scenario(task, seed=42)
        state = WorldState.from_scenario(scenario)

        print(f"\n{'=' * 60}")
        print(f"Task: {state.task_name}  |  Max Steps: {state.max_steps}")
        print(f"Reports: {state.total_reports}  |  Resources: {len(state.resources)}  |  Zones: {len(state.zones)}")
        print(f"Critical reports: {state.critical_total}")

        # Simulate a full episode with no agent actions (just advance_time)
        for step in range(state.max_steps):
            state.current_step = step
            state.advance_time()

            visible = state.get_visible_reports()
            pending = state.get_pending_reports()
            active = state.get_active_assignments()

            if state.recent_changes:
                for c in state.recent_changes:
                    print(f"  [step {step:2d}] {c}")
            if state.warnings:
                for w in state.warnings:
                    print(f"  [step {step:2d}] ⚠ {w}")

        print(f"\n  --- Episode End ---")
        print(f"  Resolved: {state.reports_resolved}")
        print(f"  Expired: {state.reports_expired}")
        print(f"  Critical missed: {state.critical_missed}")
        print(f"  False flagged: {state.reports_false_flagged}")
        print(f"  Done: {state.done}")
