"""
Disaster Relief Coordination — Deterministic Scenario Generator
================================================================
Generates reports, resources, zones, and hidden ground truth for each task.
All randomness is seeded so identical seeds produce identical scenarios.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.env.models import (
    Report,
    ReporterType,
    ReportVerdict,
    Resource,
    ResourceStatus,
    ResourceType,
    Zone,
)


# ---------------------------------------------------------------------------
# Task configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Parameters that define a task's difficulty and scope."""
    name: str
    max_steps: int
    zone_count: int
    report_count: int
    critical_count: int
    duplicate_count: int
    false_count: int
    resource_pool: List[Tuple[ResourceType, int, bool]]   # (type, capacity, can_traverse_flood)
    blockage_count: int = 0
    comms_blackout_count: int = 0
    weather_severity: int = 1
    deadline_tightness: float = 0.8     # fraction of max_steps for deadlines
    noise_rate: float = 0.2             # fraction of reports with language noise
    follow_up_count: int = 0            # number of follow-up reports
    flood_depth_max: int = 1            # max flood depth across zones

    @property
    def real_count(self) -> int:
        return self.report_count - self.duplicate_count - self.false_count


# ---------------------------------------------------------------------------
# Predefined task configs
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "task1_flood_easy": TaskConfig(
        name="task1_flood_easy",
        max_steps=12,
        zone_count=1,
        report_count=6,
        critical_count=2,
        duplicate_count=0,
        false_count=1,
        resource_pool=[
            (ResourceType.AMBULANCE, 2, False),
            (ResourceType.RESCUE_BOAT, 4, True),
            (ResourceType.SUPPLY_TRUCK, 20, False),
            (ResourceType.MEDICAL_TEAM, 3, False),
            (ResourceType.RESCUE_BOAT, 4, True),
        ],
        blockage_count=0,
        comms_blackout_count=0,
        weather_severity=2,
        deadline_tightness=0.8,
        noise_rate=0.15,
        follow_up_count=0,
        flood_depth_max=1,
    ),
    "task2_storm_medium": TaskConfig(
        name="task2_storm_medium",
        max_steps=15,
        zone_count=3,
        report_count=12,
        critical_count=4,
        duplicate_count=2,
        false_count=2,
        resource_pool=[
            (ResourceType.AMBULANCE, 2, False),
            (ResourceType.RESCUE_BOAT, 4, True),
            (ResourceType.SUPPLY_TRUCK, 20, False),
            (ResourceType.MEDICAL_TEAM, 3, False),
            (ResourceType.HELICOPTER, 2, True),
            (ResourceType.ENGINEERING_CREW, 5, False),
        ],
        blockage_count=2,
        comms_blackout_count=1,
        weather_severity=3,
        deadline_tightness=0.65,
        noise_rate=0.3,
        follow_up_count=1,
        flood_depth_max=2,
    ),
    "task3_cascade_hard": TaskConfig(
        name="task3_cascade_hard",
        max_steps=20,
        zone_count=5,
        report_count=20,
        critical_count=7,
        duplicate_count=4,
        false_count=3,
        resource_pool=[
            (ResourceType.AMBULANCE, 2, False),
            (ResourceType.RESCUE_BOAT, 4, True),
            (ResourceType.SUPPLY_TRUCK, 20, False),
            (ResourceType.MEDICAL_TEAM, 3, False),
            (ResourceType.HELICOPTER, 2, True),
            (ResourceType.ENGINEERING_CREW, 5, False),
        ],
        blockage_count=3,
        comms_blackout_count=2,
        weather_severity=5,
        deadline_tightness=0.5,
        noise_rate=0.45,
        follow_up_count=3,
        flood_depth_max=3,
    ),
}

SUPPORTED_TASKS: List[str] = list(TASK_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Zone names pool
# ---------------------------------------------------------------------------

_ZONE_NAMES = [
    "Riverside Colony",
    "Market District",
    "Old Town",
    "Industrial Area",
    "Hilltop Settlement",
    "Lakeshore Village",
    "Central Square",
    "Harbour Ward",
]

# ---------------------------------------------------------------------------
# Report text templates — intentionally messy and varied
# ---------------------------------------------------------------------------

_REPORT_CATEGORIES = [
    "flood",
    "structural_collapse",
    "medical",
    "fire",
    "road_blockage",
    "evacuation",
]

# Map category → required resource type
_CATEGORY_RESOURCE_MAP: Dict[str, ResourceType] = {
    "flood": ResourceType.RESCUE_BOAT,
    "structural_collapse": ResourceType.ENGINEERING_CREW,
    "medical": ResourceType.MEDICAL_TEAM,
    "fire": ResourceType.ENGINEERING_CREW,
    "road_blockage": ResourceType.ENGINEERING_CREW,
    "evacuation": ResourceType.SUPPLY_TRUCK,
}

# Clean report templates (used for non-noisy reports)
_CLEAN_TEMPLATES: Dict[str, List[str]] = {
    "flood": [
        "Water level rising fast at {location}. {people} people stranded on rooftop. Need immediate rescue.",
        "Flooding in {location}. Ground floor completely submerged. {people} residents trapped. Please send boats.",
        "Major flooding reported at {location}. Water entered homes, {people} people unable to evacuate. Urgent.",
        "Flash flood at {location}, water at waist height. {people} elderly people stuck. Need rescue team now.",
    ],
    "structural_collapse": [
        "Building partially collapsed at {location}. {people} people believed trapped under rubble.",
        "Wall collapse at {location}. At least {people} injured. Need engineering team and medical support.",
        "Roof caved in at the community hall, {location}. {people} people inside when it happened.",
        "Structure collapse at old warehouse in {location}. Reports of {people} workers trapped.",
    ],
    "medical": [
        "Medical emergency at {location}. {people} patients need urgent care. Local clinic overwhelmed.",
        "Multiple injuries at {location}. {people} people with serious wounds. Need ambulance and medical team.",
        "Mass casualty situation at {location}. {people} injured in stampede during evacuation.",
        "Elderly care home in {location} running out of medicine. {people} critical patients need transfer.",
    ],
    "fire": [
        "Fire broke out at {location}. {people} people evacuating but some may still be inside.",
        "Gas leak caused fire at {location}. Spreading fast. {people} families in adjacent buildings.",
        "Fire at a warehouse in {location}. Black smoke visible. {people} workers unaccounted for.",
        "Electrical fire at {location}. Area without power. {people} people need evacuation.",
    ],
    "road_blockage": [
        "Road to {location} completely blocked by debris. {people} vehicles stranded. Need clearance.",
        "Tree fell across main access road near {location}. No way in or out. {people} ambulances stuck.",
        "Landslide blocking highway near {location}. Critical supply route cut off. {people} trucks waiting.",
        "Bridge at {location} partially collapsed. Traffic halted. {people} bus passengers stranded.",
    ],
    "evacuation": [
        "Evacuation needed at {location}. {people} people in low-lying area. Water expected to rise further.",
        "Urgent evacuation request from {location}. {people} families with children and elderly. No transport.",
        "Community at {location} requesting evacuation. {people} people. Situation deteriorating rapidly.",
        "School in {location} with {people} children needs immediate bus evacuation before river overflows.",
    ],
}

# Noisy/panicked templates — typos, broken grammar, emotional
_NOISY_TEMPLATES: Dict[str, List[str]] = {
    "flood": [
        "helpp water everwhere at {location} we r on roof {people} ppl pls come fast cant swim",
        "FLOODING BAD {location}!!! {people} stuck send help NOW water coming in from doors",
        "my house {location} watwr is upto chest {people} kids here dont kno what to do plsss",
        "sos {location} flood completly cut off {people} of us no food water rising every hour help",
    ],
    "structural_collapse": [
        "BUILDING FELL DOWN at {location}!! {people} ppl inside!! send someone plsssss",
        "omg wall just collaps in {location} my neigbor {people} family stuck i can hear them screaming",
        "structure broke {location}. dunno how many. maybe {people}?? dust everywhere cant see",
        "help the roof fall at {location} {people} inside we cant get them out pls hurry",
    ],
    "medical": [
        "need doctor urgnt {location} {people} ppl bleeding bad no medicine left pls help us",
        "EMERGENCY {location} my mother and {people} others very sick no ambulance coming WHY",
        "ppl dying here {location} {people} injured cant reach hospital roads gone plss send helicopter",
        "medical crisis at {location}!! {people} patients, no doctrs no medcine water contaminated",
    ],
    "fire": [
        "FIRE FIRE {location}!! {people} ppl still inside we cant put it out help!!!",
        "thers fire at {location} gas leakng {people} families evacuating smoke everywhere cough",
        "flames at {location} spreading fast {people} kids scared need firetruck NOW pls",
        "fire and smoke {location} cant breathe {people} of us trapped on 2nd floor help",
    ],
    "road_blockage": [
        "road bloced near {location} cant get thru {people} cars stuck tree fell big one need chainsaw",
        "NO WAY OUT {location} road gone landslide {people} trucks carring supplys cant pass",
        "bridge broken at {location}!! {people} buses stranded children crying need engeneer help",
        "debris all over road {location} {people} vehicls stuck ambulnce cant get to hospital send help",
    ],
    "evacuation": [
        "NEED BUSES NOW at {location} {people} people water rising have kids and old ppl cant walk far",
        "evacuation plsss {location} {people} familys no transport landlord wont let us leave but waters coming",
        "we need to get out {location} {people} of us stranded watr rising fast no boats no trucks plsss",
        "school {location} {people} childrn still here water comng in from playground side get them out NOW",
    ],
}

# False report templates — vague, uncertain, or clearly not an emergency
_FALSE_TEMPLATES = [
    "I think I heard something weird near {location}. Not sure if it's serious. Maybe {people} people around?",
    "Someone told me there might be flooding at {location}? Haven't seen it myself. Maybe check?",
    "There used to be a problem at {location} last week. Not sure if it's still an issue. {people} people were affected then.",
    "I saw some water on the road near {location}. It's probably nothing but just in case. {people} cars drove through fine.",
    "Rumour going around that {location} is in trouble. No confirmation. {people} neighbours said they heard sirens.",
    "Not an emergency but {location} looks rough. {people} stray animals on the road. Just wanted to report.",
    "Power went out near {location}. Probably maintenance. {people} houses affected I think.",
]


# ---------------------------------------------------------------------------
# Scenario Generator
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Complete generated scenario for an episode."""
    task_config: TaskConfig
    zones: List[Zone]
    reports: List[Report]
    resources: List[Resource]
    ground_truth: Dict[str, Any]   # keyed by report_id


def generate_scenario(task_name: str, seed: int = 42) -> Scenario:
    """
    Generate a complete deterministic scenario for the given task.

    Parameters
    ----------
    task_name : str
        One of the supported task names.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Scenario
        Fully populated scenario with zones, reports, resources, and ground truth.
    """
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task '{task_name}'. Choose from {SUPPORTED_TASKS}.")

    cfg = TASK_CONFIGS[task_name]
    rng = random.Random(seed)

    zones = _generate_zones(cfg, rng)
    reports, ground_truth = _generate_reports(cfg, zones, rng)
    resources = _generate_resources(cfg, rng)

    return Scenario(
        task_config=cfg,
        zones=zones,
        reports=reports,
        resources=resources,
        ground_truth=ground_truth,
    )


# ---------------------------------------------------------------------------
# Internal generators
# ---------------------------------------------------------------------------

def _generate_zones(cfg: TaskConfig, rng: random.Random) -> List[Zone]:
    """Generate zones with varying severity, flood depth, and access conditions."""
    zone_names = rng.sample(_ZONE_NAMES, k=cfg.zone_count)
    zones: List[Zone] = []

    # Decide which zones get blockages
    blocked_indices = set(rng.sample(range(cfg.zone_count), k=min(cfg.blockage_count, cfg.zone_count)))
    # Decide which zones get comms blackouts
    blackout_indices = set(rng.sample(range(cfg.zone_count), k=min(cfg.comms_blackout_count, cfg.zone_count)))

    for i, name in enumerate(zone_names):
        zone_id = f"ZONE-{chr(65 + i)}"   # ZONE-A, ZONE-B, ...

        severity = rng.randint(1, cfg.weather_severity)
        flood_depth = rng.randint(0, cfg.flood_depth_max)
        population = rng.choice([50, 100, 200, 500, 1000, 2000])
        has_hospital = rng.random() < 0.2 if cfg.zone_count >= 3 else (i == 0 and cfg.zone_count == 1)

        access_blocked = i in blocked_indices
        blockage_clears = None
        if access_blocked:
            # Blockage clears somewhere in the middle of the episode
            blockage_clears = rng.randint(cfg.max_steps // 3, (cfg.max_steps * 2) // 3)

        comms_blackout = i in blackout_indices
        comms_restored = None
        if comms_blackout:
            comms_restored = rng.randint(cfg.max_steps // 4, cfg.max_steps // 2)

        zones.append(Zone(
            id=zone_id,
            name=name,
            severity=severity,
            access_blocked=access_blocked,
            open_incidents=0,
            blockage_clears_step=blockage_clears,
            population_density=population,
            has_hospital=has_hospital,
            flood_depth_level=flood_depth,
            last_contact_step=None,
            comms_blackout=comms_blackout,
            comms_restored_step=comms_restored,
        ))

    return zones


def _generate_reports(
    cfg: TaskConfig,
    zones: List[Zone],
    rng: random.Random,
) -> Tuple[List[Report], Dict[str, Any]]:
    """Generate all reports (real, duplicate, false) with ground truth."""

    reports: List[Report] = []
    ground_truth: Dict[str, Any] = {}
    report_idx = 1

    real_count = cfg.real_count
    total = cfg.report_count

    # --- Step 1: Generate REAL reports ---
    real_reports: List[Report] = []
    critical_assigned = 0

    for i in range(real_count):
        report_id = f"RPT-{report_idx:03d}"
        report_idx += 1

        zone = rng.choice(zones)
        category = rng.choice(_REPORT_CATEGORIES)
        resource_needed = _CATEGORY_RESOURCE_MAP[category]

        is_critical = critical_assigned < cfg.critical_count
        if is_critical:
            critical_assigned += 1

        # Auto-upgrade medical reports in hospital zones
        if category == "medical" and zone.has_hospital and not is_critical and critical_assigned < cfg.critical_count:
            is_critical = True
            critical_assigned += 1

        people_count = rng.choice([1, 2, 3, 5, 8, 12, 20, 50])

        # Decide if noisy text
        use_noise = rng.random() < cfg.noise_rate
        reporter_type = _pick_reporter_type(rng, use_noise)
        raw_text = _render_report_text(category, zone.name, people_count, use_noise, rng)

        # Stagger creation across steps (first few reports available at step 0)
        created_step = _stagger_step(i, real_count, cfg.max_steps, rng, front_load=0.4)

        # Deadline for critical reports
        deadline = None
        if is_critical:
            remaining = cfg.max_steps - created_step
            deadline = created_step + max(2, int(remaining * cfg.deadline_tightness))
            deadline = min(deadline, cfg.max_steps - 1)

        report = Report(
            id=report_id,
            raw_text=raw_text,
            zone_id=zone.id,
            category=category,
            urgency=0,  # unassessed until intake processes it
            ground_truth_verdict=ReportVerdict.REAL,
            ground_truth_duplicate_of=None,
            required_resource_type=resource_needed,
            is_critical=is_critical,
            deadline_step=deadline,
            created_step=created_step,
            reporter_type=reporter_type,
            reported_people_count=people_count,
            language_noise=use_noise,
        )
        real_reports.append(report)
        reports.append(report)
        ground_truth[report_id] = {
            "verdict": ReportVerdict.REAL.value,
            "category": category,
            "required_resource": resource_needed.value,
            "is_critical": is_critical,
            "people_count": people_count,
            "zone_id": zone.id,
        }

    # --- Step 2: Generate DUPLICATE reports ---
    for _ in range(cfg.duplicate_count):
        report_id = f"RPT-{report_idx:03d}"
        report_idx += 1

        # Pick a random real report to duplicate
        original = rng.choice(real_reports)
        zone = next(z for z in zones if z.id == original.zone_id)

        people_count = original.reported_people_count or rng.choice([2, 5, 10])
        # Duplicates are often noisier — different person calling about same thing
        use_noise = rng.random() < 0.6
        reporter_type = _pick_reporter_type(rng, use_noise)
        raw_text = _render_report_text(original.category, zone.name, people_count, use_noise, rng)

        # Duplicates arrive after the original
        created_step = min(original.created_step + rng.randint(1, 3), cfg.max_steps - 1)

        report = Report(
            id=report_id,
            raw_text=raw_text,
            zone_id=zone.id,
            category=original.category,
            urgency=0,
            ground_truth_verdict=ReportVerdict.DUPLICATE,
            ground_truth_duplicate_of=original.id,
            required_resource_type=original.required_resource_type,
            is_critical=original.is_critical,
            deadline_step=original.deadline_step,
            created_step=created_step,
            reporter_type=reporter_type,
            reported_people_count=people_count,
            language_noise=use_noise,
        )
        reports.append(report)
        ground_truth[report_id] = {
            "verdict": ReportVerdict.DUPLICATE.value,
            "duplicate_of": original.id,
            "category": original.category,
            "zone_id": zone.id,
        }

    # --- Step 3: Generate FALSE reports ---
    for _ in range(cfg.false_count):
        report_id = f"RPT-{report_idx:03d}"
        report_idx += 1

        zone = rng.choice(zones)
        people_count = rng.choice([0, 1, 2, 3])
        raw_text = _render_false_report(zone.name, people_count, rng)

        created_step = _stagger_step(
            rng.randint(0, cfg.report_count - 1),
            cfg.report_count,
            cfg.max_steps,
            rng,
            front_load=0.3,
        )

        report = Report(
            id=report_id,
            raw_text=raw_text,
            zone_id=zone.id,
            category="unknown",
            urgency=0,
            ground_truth_verdict=ReportVerdict.FALSE,
            ground_truth_duplicate_of=None,
            required_resource_type=None,
            is_critical=False,
            deadline_step=None,
            created_step=created_step,
            reporter_type=ReporterType.CITIZEN,
            reported_people_count=people_count,
            language_noise=rng.random() < 0.5,
        )
        reports.append(report)
        ground_truth[report_id] = {
            "verdict": ReportVerdict.FALSE.value,
            "category": "false_alarm",
            "zone_id": zone.id,
        }

    # --- Step 4: Generate FOLLOW-UP reports ---
    for _ in range(cfg.follow_up_count):
        report_id = f"RPT-{report_idx:03d}"
        report_idx += 1

        original = rng.choice(real_reports)
        zone = next(z for z in zones if z.id == original.zone_id)
        people_count = (original.reported_people_count or 5) + rng.randint(-2, 5)
        people_count = max(1, people_count)

        use_noise = rng.random() < cfg.noise_rate
        reporter_type = original.reporter_type
        raw_text = _render_follow_up_text(original.category, zone.name, people_count, rng)

        created_step = min(original.created_step + rng.randint(2, 5), cfg.max_steps - 1)

        report = Report(
            id=report_id,
            raw_text=raw_text,
            zone_id=zone.id,
            category=original.category,
            urgency=0,
            ground_truth_verdict=ReportVerdict.REAL,
            ground_truth_duplicate_of=None,
            required_resource_type=original.required_resource_type,
            is_critical=original.is_critical,
            deadline_step=original.deadline_step,
            created_step=created_step,
            reporter_type=reporter_type,
            reported_people_count=people_count,
            language_noise=use_noise,
            follow_up_of=original.id,
        )
        reports.append(report)
        ground_truth[report_id] = {
            "verdict": ReportVerdict.REAL.value,
            "follow_up_of": original.id,
            "category": original.category,
            "required_resource": original.required_resource_type.value if original.required_resource_type else None,
            "is_critical": original.is_critical,
            "people_count": people_count,
            "zone_id": zone.id,
        }

    # Sort all reports by created_step for deterministic ordering
    reports.sort(key=lambda r: (r.created_step, r.id))

    return reports, ground_truth


def _generate_resources(cfg: TaskConfig, rng: random.Random) -> List[Resource]:
    """Generate the resource pool from the task config."""
    resources: List[Resource] = []
    for idx, (rtype, capacity, can_flood) in enumerate(cfg.resource_pool):
        res_id = f"RES-{idx + 1:03d}"

        # Harder tasks may have fuel constraints
        fuel = None
        if cfg.weather_severity >= 4:
            fuel = rng.randint(8, cfg.max_steps)

        resources.append(Resource(
            id=res_id,
            type=rtype,
            status=ResourceStatus.AVAILABLE,
            assigned_report_id=None,
            location="base",
            eta_available_step=None,
            capacity=capacity,
            fuel_steps_remaining=fuel,
            can_traverse_flood=can_flood,
        ))

    return resources


# ---------------------------------------------------------------------------
# Text rendering helpers
# ---------------------------------------------------------------------------

def _render_report_text(
    category: str,
    location: str,
    people: int,
    noisy: bool,
    rng: random.Random,
) -> str:
    """Render a report using clean or noisy templates."""
    templates = _NOISY_TEMPLATES[category] if noisy else _CLEAN_TEMPLATES[category]
    template = rng.choice(templates)
    return template.format(location=location, people=people)


def _render_false_report(location: str, people: int, rng: random.Random) -> str:
    """Render a false/vague report."""
    template = rng.choice(_FALSE_TEMPLATES)
    return template.format(location=location, people=people)


def _render_follow_up_text(
    category: str,
    location: str,
    people: int,
    rng: random.Random,
) -> str:
    """Render a follow-up report with updated info."""
    follow_up_prefixes = [
        f"UPDATE from {location}: ",
        f"Follow-up on {location} situation: ",
        f"New info from {location}: ",
        f"Calling back about {location}: ",
    ]
    prefix = rng.choice(follow_up_prefixes)
    templates = _CLEAN_TEMPLATES[category]
    base = rng.choice(templates).format(location=location, people=people)
    return prefix + base


def _pick_reporter_type(rng: random.Random, noisy: bool) -> ReporterType:
    """Pick reporter type — noisy reports are almost always citizens."""
    if noisy:
        return rng.choices(
            [ReporterType.CITIZEN, ReporterType.FIELD_OFFICER],
            weights=[0.9, 0.1],
        )[0]
    return rng.choices(
        [ReporterType.CITIZEN, ReporterType.FIELD_OFFICER, ReporterType.AUTOMATED_SENSOR],
        weights=[0.5, 0.35, 0.15],
    )[0]


def _stagger_step(
    index: int,
    total: int,
    max_steps: int,
    rng: random.Random,
    front_load: float = 0.4,
) -> int:
    """
    Distribute reports across steps, front-loading some early.
    `front_load` fraction of reports arrive at step 0-1.
    """
    if rng.random() < front_load:
        return rng.randint(0, 1)
    else:
        return rng.randint(0, max_steps - 2)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for task in SUPPORTED_TASKS:
        scenario = generate_scenario(task, seed=42)
        cfg = scenario.task_config
        print(f"\n{'=' * 60}")
        print(f"Task: {cfg.name}")
        print(f"  Steps: {cfg.max_steps}  |  Zones: {len(scenario.zones)}  |  Reports: {len(scenario.reports)}  |  Resources: {len(scenario.resources)}")
        print(f"  Weather severity: {cfg.weather_severity}")

        for z in scenario.zones:
            flags = []
            if z.access_blocked:
                flags.append(f"BLOCKED(clears@{z.blockage_clears_step})")
            if z.comms_blackout:
                flags.append(f"COMMS_OUT(restores@{z.comms_restored_step})")
            if z.has_hospital:
                flags.append("HOSPITAL")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            print(f"  Zone {z.id} '{z.name}': severity={z.severity}, flood={z.flood_depth_level}, pop={z.population_density}{flag_str}")

        print(f"\n  Reports:")
        verdicts = {"real": 0, "duplicate": 0, "false": 0}
        critical_count = 0
        for r in scenario.reports:
            verdicts[r.ground_truth_verdict.value] += 1
            if r.is_critical:
                critical_count += 1
            noise = " [NOISY]" if r.language_noise else ""
            crit = " [CRITICAL]" if r.is_critical else ""
            dup = f" [DUP of {r.ground_truth_duplicate_of}]" if r.ground_truth_verdict == ReportVerdict.DUPLICATE else ""
            follow = f" [FOLLOWUP of {r.follow_up_of}]" if r.follow_up_of else ""
            dead = f" [deadline@{r.deadline_step}]" if r.deadline_step else ""
            print(f"    {r.id} step={r.created_step} {r.ground_truth_verdict.value:9s} {r.category:22s} zone={r.zone_id} reporter={r.reporter_type.value:16s}{crit}{noise}{dup}{follow}{dead}")
            print(f"      \"{r.raw_text[:90]}{'...' if len(r.raw_text) > 90 else ''}\"")

        print(f"\n  Verdicts: real={verdicts['real']}, duplicate={verdicts['duplicate']}, false={verdicts['false']}, critical={critical_count}")

        print(f"\n  Resources:")
        for res in scenario.resources:
            fuel = f", fuel={res.fuel_steps_remaining}" if res.fuel_steps_remaining else ""
            flood = " [FLOOD-OK]" if res.can_traverse_flood else ""
            print(f"    {res.id}: {res.type.value:20s} cap={res.capacity}{fuel}{flood}")

        # Verify determinism
        scenario2 = generate_scenario(task, seed=42)
        assert [r.id for r in scenario.reports] == [r.id for r in scenario2.reports], "DETERMINISM BROKEN!"
        assert [r.raw_text for r in scenario.reports] == [r.raw_text for r in scenario2.reports], "DETERMINISM BROKEN!"
        print(f"\n  ✓ Determinism check passed")
