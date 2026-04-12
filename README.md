---
title: Disaster Relief Coordination
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# Disaster Relief Coordination

An **OpenEnv-compatible** benchmark where an LLM agent acts as a disaster relief coordinator — triaging incident reports, dispatching scarce resources across flood-affected zones, and resolving emergencies under hard deadlines and incomplete information.

---

## Why This Environment

Real disaster response coordination is one of the hardest multi-objective reasoning problems humans face: dispatchers must **simultaneously** triage conflicting casualties, match heterogeneous resource capabilities to terrain conditions, detect misinformation under stress, and make irrevocable allocation decisions with ticking deadlines — all while new information arrives mid-crisis.

No existing OpenEnv benchmark models this **concurrent triage → dispatch → monitor loop** with terrain physics, dynamic world state, and counterfactual grading. This environment fills that gap directly:

- **For RL researchers:** A rich, multi-dimensional reward signal with 6+ grading axes — not just task completion, but *how* the agent allocates, prioritizes, and avoids mistakes.
- **For agent evaluators:** Deterministic, reproducible episodes that separate good decision-making from lucky rollouts.
- **For frontier model testing:** The hard task is structurally designed so that even a perfect-knowledge oracle scores ~0.50 — genuine frontier-model difficulty, not artificial trick questions.

---

## Multi-Agent Architecture

The coordinator agent (the LLM) orchestrates three specialist sub-agents through **tool-mediated delegation**. Each sub-agent encapsulates a domain-specific workflow:

```
                    ┌──────────────────────┐
                    │   LLM COORDINATOR    │
                    │  (decision-maker)    │
                    └──────┬───────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
    ┌─────────────┐ ┌────────────┐  ┌──────────────┐
    │ INTAKE AGENT│ │DISPATCH    │  │MONITOR AGENT │
    │             │ │AGENT       │  │              │
    │• classify   │ │• inventory │  │• check status│
    │• urgency    │ │• send      │  │• close case  │
    │• verify     │ │• reroute   │  │• flag false  │
    └─────────────┘ └────────────┘  └──────────────┘
```

**The coordinator decides at each step:** Should I triage the new report, dispatch a resource to a critical incident, flag a suspected false alarm, or monitor an ongoing operation? This mirrors how real emergency operations centers work — a single decision-maker delegating to specialized teams while maintaining situational awareness.

### Decision Flow Per Step

```
1. Observe: Read pending reports, zone conditions, resource status, warnings
2. Prioritize: Which action has the highest marginal value right now?
   - Critical deadline approaching? → Dispatch immediately
   - Unclassified report? → Send to intake agent
   - Low-confidence report? → Flag as false alarm
   - Stuck resource? → Reroute
   - Active assignment aging? → Monitor/close
3. Act: Issue one tool call ({"tool": "...", "args": {...}})
4. World advances: Assignments progress, deadlines tick, conditions change
5. Repeat until max_steps
```

---

## What Makes It Hard

| Challenge | Mechanic | Why it matters |
|---|---|---|
| **Terrain constraints** | Flood depth ≥ 2 requires boats or helicopters — sending an ambulance fails silently | Forces resource-type reasoning |
| **False alarm detection** | Some reports are fabricated or duplicates; flagging correctly scores points, missing them wastes resources | Tests information filtering under uncertainty |
| **Hard deadlines** | Critical reports expire after N steps; letting them expire is heavily penalized by the counterfactual grader | Punishes slow, overly cautious strategies |
| **Resource scarcity** | 6 resources for 10-17 reports across 1-5 zones | Forces triage — the agent *cannot* help everyone |
| **Dynamic conditions** | Comms blackouts delay reports; road blockages strand resources; both clear mid-episode | Rewards adaptive replanning |
| **Follow-up reports** | New reports arrive mid-episode (task2: +1, task3: +3) | Tests reactive planning, not just static optimization |
| **Counterfactual grading** | The grader checks if a correct resource was idle when a critical report expired | Catches agents that are busy but inefficient |

---

## Observation Space

Each step delivers a structured JSON snapshot of the world — no hidden state, no tricks:

| Field | Type | Description |
|---|---|---|
| `step` / `max_steps` | int | Episode progress |
| `situation_brief` | string | Crisis narrative generated at step 0 — sets the scene for the LLM |
| `pending_reports` | array | Outstanding incidents with urgency, deadline, zone, category, verification status |
| `active_assignments` | array | In-progress dispatches with ETA, route status, stuck flag |
| `available_resources` | array | Full fleet — type, availability, flood capability |
| `zones` | array | Per-zone flood depth, access status, comms status, severity |
| `recent_changes` | array | Delta from previous step (arrivals, expirations, blockage clearances) |
| `warnings` | array | Deadline alerts, stuck resources, monitoring nudges |
| `available_tools` | array | Full tool signatures with descriptions — the agent can read this |
| `last_action_result` | string | Feedback from the previous tool call |
| `last_action_error` | string | Validation error if the action was illegal |

---

## Action Space — 12 Tools

Every action is a JSON tool call:

```json
{"tool": "<tool_name>", "args": {"key": "value"}}
```

### Coordinator Delegation Tools
| Tool | Args | What it does |
|---|---|---|
| `call_intake_agent` | `report_id` | Full triage pipeline: classify → assess urgency → verify authenticity (3 operations, 1 step) |
| `call_dispatch_agent` | `resource_id, report_id` | Check inventory then dispatch — safer than raw `send_resource` |
| `call_monitor_agent` | `target_id` | Monitor an assignment; auto-closes if resolved |

### Direct Tools
| Tool | Args | Purpose |
|---|---|---|
| `classify_report` | `report_id` | Classify report category only |
| `assess_report_urgency` | `report_id` | Score urgency (0-10) |
| `verify_report` | `report_id` | Authenticate report; reveals confidence score |
| `send_resource` | `resource_id, report_id` | Directly dispatch a resource |
| `get_resources` | *(none)* | Enumerate full fleet with status |
| `reroute_resource` | `resource_id` | Reroute a stuck or misassigned resource |
| `check_operation` | `target_id` | Poll report or assignment status |
| `close_case` | `report_id` | Close a resolved case |
| `mark_false_report` | `report_id, reason` | Flag as false alarm / duplicate |

---

## Tasks and Difficulty Progression

Three tasks form a clear difficulty ladder. Complexity scales across **every axis simultaneously** — more zones, more reports, more deception, stricter deadlines, and deeper grading:

| Task | Diff | Steps | Zones | Reports | Critical | False/Dup | Resources | Grader Axes |
|---|---|---|---|---|---|---|---|---|
| `task1_flood_easy` | Easy | 12 | 1 | 6 | 2 | 1 | 5 | 3 |
| `task2_storm_medium` | Medium | 20 | 3 | 11 | 3 | 3 | 6 | 5 |
| `task3_cascade_hard` | Hard | 30 | 5 | 17 | 4 | 5 | 6 | 6 |

All tasks are generated **deterministically from a seed** — identical seeds produce identical scenarios.

### Resource Fleet

| Type | Flood-capable | Role |
|---|---|---|
| Ambulance | No | Medical transport |
| Rescue Boat | **Yes** | Flood zone rescue |
| Supply Truck | No | Evacuation / supplies |
| Medical Team | No | On-site medical care |
| Helicopter | **Yes** | All-terrain rapid response |
| Engineering Crew | No | Structural / road repair |

---

## Grading — Multi-Dimensional, Exploit-Resistant

Each episode is scored on **[0.0, 1.0]**. The grader evaluates multiple complementary dimensions, making it robust against degenerate strategies (dispatching everything wildly fails on resource correctness; ignoring false alarms fails on F1; playing it safe fails on counterfactual).

### Task 1 — Flood Response (Easy)
*40% resolution + 30% critical + 30% efficiency*

| Dimension | What it measures |
|---|---|
| Resolution rate | Fraction of real reports resolved |
| Critical resolution | Critical reports resolved before deadline |
| Efficiency | Penalizes wasted dispatches to false/duplicate reports |

### Task 2 — Multi-Zone Storm (Medium)
*30% resolution + 25% critical + 15% F1 + 15% resource match + 15% counterfactual*

| Dimension | What it measures |
|---|---|
| Resolution rate | Coverage across zones |
| Critical resolution | Deadline adherence under comms blackout |
| False alarm F1 | Balanced precision × recall for flagged false reports |
| Resource match | Whether dispatch used the correct resource type for the incident |
| Counterfactual | Penalty when critical report expired but a valid resource was idle |

### Task 3 — Cascade Disaster (Hard)
*30% resolution + 25% critical + 15% F1 + 10% resource match + 10% monitoring + 10% counterfactual*

| Dimension | What it measures |
|---|---|
| Resolution rate | Coverage across 5 zones with blockages and blackouts |
| Critical resolution | Hard deadlines with deep flooding |
| False alarm F1 | Precision and recall balanced |
| Resource match | Flood-capability and type alignment |
| Monitoring quality | Did the agent check and close assignments promptly? |
| Counterfactual | Missed opportunities despite available resources |

### Reward Shaping

The per-step reward provides a **dense, informative training signal** — not just sparse end-of-episode feedback:

- **Base:** +0.1 per step (minimal, avoids reward gaming)
- **Triage rewards:** +1.0 classify, +1.0 urgency, +0.5 verify via `call_intake_agent`
- **Dispatch:** +2.0 correct resource type, -1.0 wrong type, -1.0 dispatching to false alarm
- **False alarm flagging:** +1.0 correct flag, -1.5 flagging a real report, -2.0 flagging a critical report
- **Temporal urgency multiplier:** 2.0x → 1.0x bonus for early action on deadline reports
- **Penalties:** -0.5 for malformed actions, errors, or repeating the same action twice

---

## Score Calibration

| Agent | task1 (easy) | task2 (medium) | task3 (hard) |
|---|---|---|---|
| Random / do-nothing | ~0.00 | ~0.00 | ~0.00 |
| Heuristic baseline | **0.46** | **0.79** | **0.48** |
| Oracle (perfect knowledge) | 0.71 | 0.79 | 0.50 |
| Theoretical maximum | 1.00 | 1.00 | 1.00 |

The heuristic baseline is flood-aware and type-matched, so it performs well on medium difficulty. The hard task's score ceiling (~0.50 even for the oracle) reflects genuine structural difficulty — deep flooding, late-arriving reports, and resource scarcity make perfect performance impossible. This leaves meaningful headroom for frontier LLMs to demonstrate planning and reasoning.

---

## Environment Design

Clean software boundaries with no shared mutable state between episodes:

```
DisasterReliefEnv
├── reset(task, seed)     → StepResult   # deterministic, clean state
├── step(action)          → StepResult   # {observation, reward, done, info}
├── grade()               → GradeResult  # end-of-episode score breakdown
└── close()               # cleanup
```

- **State management:** `WorldState` encapsulates all simulation state. `advance_time()` runs the simulation tick each step — resource ETAs decrement, deadlines expire, roads unblock, comms restore, fuel depletes.
- **Observation builder:** Stateless function that produces a clean dict from `WorldState` — no internal references leak.
- **Tool registry:** All 12 tools registered in `tool_registry.py` — adding a tool requires touching one file.
- **Reward:** Pure function over `(state, action_result)` — no side effects, fully unit-testable.
- **Graders:** Deterministic, reproducible, score clamped to [0.0, 1.0]. Same seed + same actions = same score.

---

## API Endpoints

Served as a stateful HTTP API (FastAPI, port 7860):

| Method | Path | Body / Response |
|---|---|---|
| `GET` | `/` | Health check → `{"status": "ok"}` |
| `GET` | `/tasks` | List task names |
| `POST` | `/reset` | `{"task_name": "...", "seed": 42}` → `StepResult` |
| `POST` | `/step` | `{"tool": "...", "args": {...}}` → `StepResult` |
| `GET` | `/state` | Internal world state snapshot |
| `POST` | `/grade` | Episode grade → `GradeResult` |

---

## Code Structure

```
├── app.py                    # FastAPI server — OpenEnv HTTP API
├── inference.py              # Baseline agent: LLM + heuristic fallback
├── openenv.yaml              # OpenEnv spec
├── Dockerfile                # Python 3.11-slim, non-root, port 7860
├── requirements.txt
└── src/env/
    ├── environment.py        # DisasterReliefEnv — reset / step / grade
    ├── models.py             # Pydantic v2: 7 enums, 4 domain types
    ├── state.py              # WorldState + advance_time() simulation
    ├── scenarios.py          # Deterministic scenario generator (seeded RNG)
    ├── observation.py        # Observation builder (stateless)
    ├── rewards.py            # Per-step reward function (pure, structured)
    ├── graders.py            # Episode graders: F1, counterfactual, resource match
    ├── tool_registry.py      # Tool dispatch table + validation
    ├── tools_intake.py       # classify / urgency / verify
    ├── tools_dispatch.py     # inventory / send / reroute
    ├── tools_monitor.py      # check / close / flag false
    └── tools_coordinator.py  # Sub-agent delegation wrappers
```

All domain models use Pydantic v2 with full type annotations. The test suite covers environment logic, grader bounds + idempotency, server endpoints, scenario determinism, stdout compliance, and LLM integration.

---

## Running Locally

**Requirements:** Python 3.11+, or Docker.

```bash
# Install
pip install -r requirements.txt

# Start server
uvicorn app:app --host 0.0.0.0 --port 7860

# Heuristic baseline (no API key needed)
python inference.py --heuristic-only

# With LLM
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
python inference.py
```

**Docker:**
```bash
docker build -t disaster-relief .
docker run -p 7860:7860 -e HF_TOKEN="your-token" disaster-relief
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | **Yes** | *(none)* | Bearer token for LLM API |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `PORT` | No | `7860` | Server port |
