"""
Disaster Relief Coordination — Intake Tools
=============================================
Pure functions that read/write WorldState for report triage.
Each returns (result_text, reward_delta).
"""

from __future__ import annotations

from typing import Tuple

from src.env.models import ReportStatus, ReportVerdict, ReporterType
from src.env.state import WorldState


# Map category keywords that might appear in raw text → canonical category
_KEYWORD_MAP = {
    "flood": "flood",
    "water": "flood",
    "submerge": "flood",
    "drown": "flood",
    "collaps": "structural_collapse",
    "rubble": "structural_collapse",
    "roof": "structural_collapse",
    "cave": "structural_collapse",
    "structur": "structural_collapse",
    "medic": "medical",
    "injur": "medical",
    "bleed": "medical",
    "patient": "medical",
    "doctor": "medical",
    "ambulan": "medical",
    "hospital": "medical",
    "fire": "fire",
    "flame": "fire",
    "smoke": "fire",
    "burn": "fire",
    "block": "road_blockage",
    "debris": "road_blockage",
    "landslide": "road_blockage",
    "bridge": "road_blockage",
    "tree fell": "road_blockage",
    "evacuat": "evacuation",
    "bus": "evacuation",
    "transport": "evacuation",
    "get out": "evacuation",
}

_CATEGORY_RESOURCE = {
    "flood": "rescue_boat",
    "structural_collapse": "engineering_crew",
    "medical": "medical_team",
    "fire": "engineering_crew",
    "road_blockage": "engineering_crew",
    "evacuation": "supply_truck",
}


def classify_report(state: WorldState, report_id: str) -> Tuple[str, float]:
    """
    Classify a report's category based on its raw text.

    Sets report.category, report.classified = True.
    Returns (result_text, reward_delta).
    """
    report = state.get_report(report_id)
    if report is None:
        return f"Error: Report '{report_id}' not found.", -0.5

    if report.status not in (ReportStatus.PENDING, ReportStatus.TRIAGED):
        return f"Report {report_id} is already {report.status.value} — cannot classify.", 0.0

    if report.classified:
        return f"Report {report_id} has already been classified as '{report.category}'.", 0.0

    # Determine category from raw text keyword matching
    text_lower = report.raw_text.lower()
    detected_category = "unknown"
    for keyword, category in _KEYWORD_MAP.items():
        if keyword in text_lower:
            detected_category = category
            break

    report.category = detected_category
    report.classified = True

    # Compute reward: compare to ground truth category
    gt = state.ground_truth.get(report_id, {})
    gt_category = gt.get("category", "unknown")
    reward = 0.0

    if gt_category == "false_alarm":
        # Classifying a false alarm — no reward or penalty for category guess
        result = (
            f"Report {report_id} classified as '{detected_category}'. "
            f"Zone: {report.zone_id}. Suggested resource: {_CATEGORY_RESOURCE.get(detected_category, 'unknown')}."
        )
    elif detected_category == gt_category:
        reward = 1.0
        result = (
            f"Report {report_id} classified as '{detected_category}' (correct). "
            f"Zone: {report.zone_id}. Suggested resource: {_CATEGORY_RESOURCE.get(detected_category, 'unknown')}."
        )
    else:
        reward = 0.0  # no penalty for wrong category — just no bonus
        result = (
            f"Report {report_id} classified as '{detected_category}'. "
            f"Zone: {report.zone_id}. Suggested resource: {_CATEGORY_RESOURCE.get(detected_category, 'unknown')}."
        )

    # Promote to TRIAGED if still PENDING
    if report.status == ReportStatus.PENDING:
        report.status = ReportStatus.TRIAGED

    return result, reward


def assess_report_urgency(state: WorldState, report_id: str) -> Tuple[str, float]:
    """
    Assess a report's urgency on a 0-10 scale using heuristics on the raw text
    and metadata.

    Sets report.urgency, report.urgency_assessed = True.
    Returns (result_text, reward_delta).
    """
    report = state.get_report(report_id)
    if report is None:
        return f"Error: Report '{report_id}' not found.", -0.5

    if report.status not in (ReportStatus.PENDING, ReportStatus.TRIAGED):
        return f"Report {report_id} is already {report.status.value} — cannot assess urgency.", 0.0

    if report.urgency_assessed:
        return f"Report {report_id} urgency already assessed: {report.urgency}/10.", 0.0

    # Heuristic urgency scoring
    score = 3  # baseline

    text_lower = report.raw_text.lower()

    # Panic indicators
    panic_words = ["help", "sos", "urgent", "emergency", "dying", "critical", "trapped", "stuck", "now", "fast"]
    panic_count = sum(1 for w in panic_words if w in text_lower)
    score += min(panic_count, 3)

    # People count
    if report.reported_people_count is not None:
        if report.reported_people_count >= 20:
            score += 2
        elif report.reported_people_count >= 5:
            score += 1

    # Reporter credibility
    if report.reporter_type == ReporterType.FIELD_OFFICER:
        score += 1
    elif report.reporter_type == ReporterType.AUTOMATED_SENSOR:
        score += 1

    # Zone severity
    zone = state.get_zone(report.zone_id)
    if zone:
        if zone.severity >= 4:
            score += 1
        if zone.flood_depth_level >= 2:
            score += 1
        if zone.has_hospital and report.category == "medical":
            score += 1

    # Deadline proximity
    if report.deadline_step is not None:
        remaining = report.deadline_step - state.current_step
        if remaining <= 2:
            score += 2
        elif remaining <= 4:
            score += 1

    score = max(1, min(10, score))

    report.urgency = score
    report.urgency_assessed = True

    # Promote to TRIAGED if still PENDING
    if report.status == ReportStatus.PENDING:
        report.status = ReportStatus.TRIAGED

    # Reward: compare with ground truth criticality
    gt = state.ground_truth.get(report_id, {})
    is_critical = gt.get("is_critical", False)
    reward = 0.0

    if is_critical and score >= 7:
        reward = 1.0  # correctly flagged as high urgency
    elif not is_critical and score <= 5:
        reward = 0.5  # reasonable assessment for non-critical
    elif is_critical and score < 5:
        reward = -0.5  # underestimated a critical report

    result = (
        f"Report {report_id} urgency assessed: {score}/10. "
        f"{'CRITICAL — prioritize immediately!' if score >= 8 else 'High priority.' if score >= 6 else 'Moderate priority.' if score >= 4 else 'Low priority.'}"
    )
    if report.deadline_step is not None:
        remaining = report.deadline_step - state.current_step
        result += f" Deadline in {remaining} step(s)."

    return result, reward


def verify_report(state: WorldState, report_id: str) -> Tuple[str, float]:
    """
    Verify a report's authenticity: real, duplicate, or false alarm.

    Sets report.verified, report.verification_confidence.
    Returns (result_text, reward_delta).
    """
    report = state.get_report(report_id)
    if report is None:
        return f"Error: Report '{report_id}' not found.", -0.5

    if report.status not in (ReportStatus.PENDING, ReportStatus.TRIAGED):
        return f"Report {report_id} is already {report.status.value} — cannot verify.", 0.0

    if report.verified:
        return (
            f"Report {report_id} already verified: "
            f"confidence={report.verification_confidence:.0%} genuine.",
            0.0,
        )

    # Deterministic verification based on ground truth + noise
    gt = state.ground_truth.get(report_id, {})
    verdict = gt.get("verdict", "real")

    if verdict == "real":
        confidence = 0.85 if report.language_noise else 0.95
        if report.reporter_type == ReporterType.FIELD_OFFICER:
            confidence = min(1.0, confidence + 0.05)
        elif report.reporter_type == ReporterType.AUTOMATED_SENSOR:
            confidence = min(1.0, confidence + 0.08)
    elif verdict == "duplicate":
        confidence = 0.40  # moderate — looks real but is a duplicate
    else:  # false
        confidence = 0.20  # low — vague language triggers suspicion

    report.verified = True
    report.verification_confidence = confidence

    # Promote to TRIAGED if still PENDING
    if report.status == ReportStatus.PENDING:
        report.status = ReportStatus.TRIAGED

    # Build result
    if confidence >= 0.7:
        assessment = "LIKELY GENUINE"
    elif confidence >= 0.4:
        assessment = "UNCERTAIN — possible duplicate or exaggeration"
    else:
        assessment = "SUSPICIOUS — likely false alarm or vague report"

    result = f"Report {report_id} verification: {assessment} (confidence={confidence:.0%})."

    if verdict == "duplicate":
        dup_of = gt.get("duplicate_of", "unknown")
        result += f" May be a duplicate of report {dup_of}."

    # Reward for verification is given when the agent acts on it
    # (mark_false_report or send_resource), not here — just informational
    reward = 0.5  # small reward for doing due diligence

    return result, reward
