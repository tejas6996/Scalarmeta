"""
Disaster Relief Coordination — Core Pydantic Models
=====================================================
Every data structure used across the environment is defined here.
All models are immutable-friendly but mutable where state changes are needed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportStatus(str, Enum):
    """Lifecycle status of an incoming disaster report."""
    PENDING = "pending"                # Arrived but not yet triaged
    TRIAGED = "triaged"                # Intake processing done
    DISPATCHED = "dispatched"          # Resource assigned and en-route
    RESOLVED = "resolved"              # Incident closed successfully
    FALSE = "false"                    # Marked as false alarm / duplicate
    EXPIRED = "expired"                # Deadline passed without resolution


class ReportVerdict(str, Enum):
    """Hidden ground truth about a report's authenticity."""
    REAL = "real"
    DUPLICATE = "duplicate"
    FALSE = "false"


class ReporterType(str, Enum):
    """Who submitted the report — affects credibility."""
    CITIZEN = "citizen"
    FIELD_OFFICER = "field_officer"
    AUTOMATED_SENSOR = "automated_sensor"


class ResourceType(str, Enum):
    """Types of deployable resources."""
    AMBULANCE = "ambulance"
    RESCUE_BOAT = "rescue_boat"
    SUPPLY_TRUCK = "supply_truck"
    MEDICAL_TEAM = "medical_team"
    HELICOPTER = "helicopter"
    ENGINEERING_CREW = "engineering_crew"


class ResourceStatus(str, Enum):
    """Current availability of a resource unit."""
    AVAILABLE = "available"
    DEPLOYED = "deployed"
    RETURNING = "returning"


class RouteStatus(str, Enum):
    """Route condition for an active assignment."""
    CLEAR = "clear"
    BLOCKED = "blocked"
    REROUTED = "rerouted"


class AssignmentStatus(str, Enum):
    """Status of a resource-to-report assignment."""
    EN_ROUTE = "en_route"
    ON_SITE = "on_site"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Core Domain Models
# ---------------------------------------------------------------------------

class Report(BaseModel):
    """A disaster incident report received by the coordination center."""

    id: str = Field(..., description="Unique report identifier, e.g. 'RPT-001'.")
    raw_text: str = Field(..., description="The raw text of the incoming report.")
    zone_id: str = Field(..., description="Zone where the incident is located.")
    category: str = Field(
        default="unknown",
        description="Report category: 'flood', 'structural_collapse', 'medical', 'fire', 'road_blockage', 'evacuation', 'unknown'.",
    )
    urgency: int = Field(
        default=0, ge=0, le=10,
        description="Urgency score 0-10. 0=unassessed, 1=low, 10=critical.",
    )

    # --- Hidden ground truth (not exposed to the agent) ---
    ground_truth_verdict: ReportVerdict = Field(
        ..., description="Whether this report is real, a duplicate, or false."
    )
    ground_truth_duplicate_of: Optional[str] = Field(
        default=None,
        description="If duplicate, the ID of the original report.",
    )
    required_resource_type: Optional[ResourceType] = Field(
        default=None,
        description="The correct resource type needed (None for false reports).",
    )
    is_critical: bool = Field(
        default=False,
        description="Whether this is a critical-priority incident (lives at immediate risk).",
    )
    deadline_step: Optional[int] = Field(
        default=None,
        description="Step by which this must be resolved, or a penalty is incurred. None = no deadline.",
    )

    # --- Mutable state ---
    status: ReportStatus = Field(
        default=ReportStatus.PENDING,
        description="Current lifecycle status.",
    )
    created_step: int = Field(
        ..., description="The step at which this report becomes visible to the agent.",
    )
    resolved_step: Optional[int] = Field(
        default=None, description="Step at which the report was resolved/closed.",
    )
    assigned_resource_id: Optional[str] = Field(
        default=None, description="ID of the resource currently assigned to this report.",
    )

    # --- Enriched metadata ---
    reporter_type: ReporterType = Field(
        default=ReporterType.CITIZEN,
        description="Who submitted the report (citizen, field_officer, automated_sensor).",
    )
    reported_people_count: Optional[int] = Field(
        default=None, description="Number of people mentioned in the report (None = not stated).",
    )
    language_noise: bool = Field(
        default=False,
        description="Whether the text has typos, partial info, or panicked language.",
    )
    follow_up_of: Optional[str] = Field(
        default=None,
        description="If this is a follow-up report, the ID of the original report.",
    )

    # --- Intake classification results (filled by tools) ---
    classified: bool = Field(default=False, description="Whether intake classification has been run.")
    urgency_assessed: bool = Field(default=False, description="Whether urgency assessment has been run.")
    verified: bool = Field(default=False, description="Whether verification has been run.")
    verification_confidence: Optional[float] = Field(
        default=None, description="Confidence 0.0-1.0 that the report is genuine.",
    )


class Resource(BaseModel):
    """A deployable resource unit (vehicle, team, etc.)."""

    id: str = Field(..., description="Unique resource identifier, e.g. 'RES-001'.")
    type: ResourceType = Field(..., description="Type of this resource.")
    status: ResourceStatus = Field(
        default=ResourceStatus.AVAILABLE,
        description="Current availability status.",
    )
    assigned_report_id: Optional[str] = Field(
        default=None, description="Report ID this resource is currently assigned to.",
    )
    location: str = Field(
        default="base", description="Current location identifier (zone_id or 'base').",
    )
    eta_available_step: Optional[int] = Field(
        default=None,
        description="Step at which this resource becomes available again (None = already available).",
    )
    capacity: int = Field(
        default=1, ge=1,
        description="How many people/units this resource can serve (e.g. ambulance=2, truck=20).",
    )
    fuel_steps_remaining: Optional[int] = Field(
        default=None,
        description="Steps of fuel remaining before needing to return to base (None = unlimited).",
    )
    can_traverse_flood: bool = Field(
        default=False,
        description="Whether this resource can operate in flooded zones (boats, helicopters).",
    )


class Zone(BaseModel):
    """A geographic zone in the disaster area."""

    id: str = Field(..., description="Unique zone identifier, e.g. 'ZONE-A'.")
    name: str = Field(..., description="Human-readable zone name.")
    severity: int = Field(
        default=1, ge=1, le=5,
        description="Zone severity level 1-5. Higher = worse conditions.",
    )
    access_blocked: bool = Field(
        default=False,
        description="Whether road access to this zone is currently blocked.",
    )
    open_incidents: int = Field(
        default=0, ge=0,
        description="Number of currently open (unresolved) incidents in this zone.",
    )
    blockage_clears_step: Optional[int] = Field(
        default=None,
        description="Step at which the blockage clears (None = no blockage or permanent).",
    )
    population_density: int = Field(
        default=100, ge=0,
        description="Approximate population in this zone. Affects urgency weighting.",
    )
    has_hospital: bool = Field(
        default=False,
        description="Whether a hospital is in this zone — auto-upgrades medical reports to critical.",
    )
    flood_depth_level: int = Field(
        default=0, ge=0, le=3,
        description="Flood depth: 0=dry, 1=shallow(vehicles ok), 2=moderate(boats needed), 3=deep(helicopter only).",
    )
    last_contact_step: Optional[int] = Field(
        default=None,
        description="Last step a report was received from this zone (None = never contacted).",
    )
    comms_blackout: bool = Field(
        default=False,
        description="Whether communication is currently blacked out in this zone.",
    )
    comms_restored_step: Optional[int] = Field(
        default=None,
        description="Step at which communications are restored (None = no blackout or permanent).",
    )


class Assignment(BaseModel):
    """Tracks a resource dispatched to a report."""

    id: str = Field(..., description="Unique assignment identifier, e.g. 'ASG-001'.")
    resource_id: str = Field(..., description="ID of the assigned resource.")
    report_id: str = Field(..., description="ID of the target report.")
    created_step: int = Field(..., description="Step when the assignment was created.")
    route_status: RouteStatus = Field(
        default=RouteStatus.CLEAR,
        description="Current route condition.",
    )
    status: AssignmentStatus = Field(
        default=AssignmentStatus.EN_ROUTE,
        description="Current assignment status.",
    )
    expected_completion_step: int = Field(
        ..., description="Step by which the assignment should be completed.",
    )
    stuck: bool = Field(
        default=False,
        description="Whether the assignment is stuck/delayed beyond expected completion.",
    )


# ---------------------------------------------------------------------------
# Observation — What the LLM coordinator sees each step
# ---------------------------------------------------------------------------

class ReportSummary(BaseModel):
    """Compact report info exposed in observations (no hidden ground truth)."""
    id: str
    raw_text: str
    zone_id: str
    category: str
    urgency: int
    status: ReportStatus
    is_critical: bool
    created_step: int
    deadline_step: Optional[int] = None
    assigned_resource_id: Optional[str] = None
    classified: bool = False
    urgency_assessed: bool = False
    verified: bool = False
    verification_confidence: Optional[float] = None
    reporter_type: ReporterType = ReporterType.CITIZEN
    reported_people_count: Optional[int] = None
    language_noise: bool = False


class ResourceSummary(BaseModel):
    """Compact resource info exposed in observations."""
    id: str
    type: ResourceType
    status: ResourceStatus
    assigned_report_id: Optional[str] = None
    location: str = "base"


class AssignmentSummary(BaseModel):
    """Compact assignment info exposed in observations."""
    id: str
    resource_id: str
    report_id: str
    created_step: int
    route_status: RouteStatus
    status: AssignmentStatus
    expected_completion_step: int
    stuck: bool = False


class Observation(BaseModel):
    """The full observation payload the LLM coordinator receives each step."""

    step: int = Field(..., description="Current step number (0-based).")
    max_steps: int = Field(..., description="Total steps in this episode.")
    task_name: str = Field(..., description="Name of the current task.")

    pending_reports: List[ReportSummary] = Field(
        default_factory=list,
        description="Reports that are visible and not yet resolved.",
    )
    active_assignments: List[AssignmentSummary] = Field(
        default_factory=list,
        description="Currently active resource assignments.",
    )
    available_resources: List[ResourceSummary] = Field(
        default_factory=list,
        description="Resources and their current status.",
    )

    recent_changes: List[str] = Field(
        default_factory=list,
        description="What changed since the last step (new reports, completions, failures, etc.).",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Urgent warnings: approaching deadlines, stuck assignments, empty inventory, etc.",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="List of tool names the agent can call this step.",
    )

    last_action_result: Optional[str] = Field(
        default=None,
        description="Result/feedback from the previous action, if any.",
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message if the previous action was malformed or invalid.",
    )
    weather_severity: int = Field(
        default=1, ge=1, le=5,
        description="Current overall weather severity 1-5. Affects zone conditions.",
    )
    situation_brief_submitted: bool = Field(
        default=False,
        description="Whether the coordinator has submitted an initial situation assessment.",
    )


# ---------------------------------------------------------------------------
# Action — What the LLM coordinator outputs each step
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """A single tool-call action from the coordinator."""

    tool: str = Field(
        ...,
        description="Name of the tool to invoke (e.g. 'call_intake_agent', 'send_resource').",
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool call.",
    )


# ---------------------------------------------------------------------------
# Reward — Structured step reward breakdown
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Per-step reward breakdown."""

    total: float = Field(default=0.0, description="Net reward for this step.")
    base: float = Field(default=0.5, description="Base survival reward per step.")
    triage_reward: float = Field(default=0.0, description="Reward for correct triage actions.")
    dispatch_reward: float = Field(default=0.0, description="Reward for correct dispatch actions.")
    monitor_reward: float = Field(default=0.0, description="Reward for correct monitoring actions.")
    penalty: float = Field(default=0.0, description="Negative penalties (malformed, wrong type, missed deadline, etc.).")


# ---------------------------------------------------------------------------
# Step Result — Returned by env.step()
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Full result from a single environment step."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment State — Public snapshot for /state endpoint
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Public environment snapshot (no hidden ground truth)."""

    current_step: int
    max_steps: int
    task_name: str
    total_reward: float
    total_reports: int
    reports_resolved: int
    reports_expired: int
    reports_false_flagged: int
    critical_reports_total: int
    critical_reports_missed: int
    active_assignments: int
    resources_available: int
    resources_deployed: int


# ---------------------------------------------------------------------------
# Episode Memory Entry
# ---------------------------------------------------------------------------

class EpisodeMemoryEntry(BaseModel):
    """A single record in the episode's working memory."""

    step: int = Field(..., description="Step number when this entry was created.")
    actor: str = Field(
        default="coordinator",
        description="Who performed this action: 'coordinator', 'intake', 'dispatch', 'monitor'.",
    )
    tool_name: str = Field(..., description="Name of the tool that was called.")
    input_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool.")
    summary: str = Field(default="", description="Human-readable summary of what happened.")
    result_status: str = Field(
        default="success",
        description="Outcome: 'success', 'error', 'partial', 'no_effect'.",
    )
    reward: float = Field(default=0.0, description="Reward earned/lost by this action.")
    important_entities: List[str] = Field(
        default_factory=list,
        description="IDs of reports, resources, or assignments involved.",
    )
