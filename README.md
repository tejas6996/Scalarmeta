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

An **OpenEnv-compatible** benchmark where an LLM agent acts as a disaster relief coordinator — triaging incident reports, allocating scarce resources across flood-affected zones, and resolving emergencies under hard deadlines and incomplete information.

---

## Overview

Disaster response coordination is a genuinely hard, high-stakes real-world problem. Dispatchers must reason under time pressure, prioritise conflicting casualties, match resource capabilities to terrain conditions, and identify false alarms without dismissing genuine emergencies. This environment models that exact challenge as a structured, reproducible benchmark.

The environment is designed to be useful for:
- Evaluating whether LLMs can reason about **resource constraints, causality, and urgency** simultaneously
- Training agents on **multi-step tool use** with real consequences for wrong decisions
- Benchmarking **planning vs. reactive** strategies on progressively harder scenarios

---

## What the Agent Does

Each episode, the agent manages an active disaster area. Per step, it selects one tool to call — triaging a new report, dispatching a resource, monitoring an active operation, or flagging a false alarm. The world advances after each action: assignments progress, deadlines tick down, comms return, and roads unblock.

**The agent must reason about:**

| Challenge | Detail |
|---|---|
| Terrain constraints | Flood depth ≥ 2 requires flood-capable resources (rescue boats, helicopters); sending an ambulance fails |
| False alarm detection | Some reports are fabricated or duplicate; marking them correctly improves score, missing them wastes resources |
| Hard deadlines | Critical reports expire after a fixed number of steps; letting them expire is heavily penalised |
| Resource scarcity | Multiple reports compete for a small, heterogeneous fleet |
| Dynamic conditions | Comms blackouts and road blockages clear mid-episode; the agent must adapt |

**Resource fleet:**

| Type | Flood-capable | Capacity | Role |
|---|---|---|---|
| Ambulance | No | 2 | Medical transport |
| Rescue Boat | Yes | 4 | Flood rescue |
| Supply Truck | No | 20 | Evacuation / supplies |
| Medical Team | No | 3 | On-site care |
| Helicopter | Yes | 2 | All-terrain rapid response |
| Engineering Crew | No | 5 | Structural / road repair |

---

## Observation Space

At each step the agent receives a full structured snapshot of the world state:

| Field | Type | Description |
|---|---|---|
| `step` / `max_steps` | int | Episode progress |
| `pending_reports` | array | Outstanding incidents with urgency, deadline, zone, and verification status |
| `active_assignments` | array | In-progress dispatches with ETA |
| `available_resources` | array | Full fleet status — type, availability, flood capability |
| `zones` | array | Per-zone flood depth, access status, comms status |
| `recent_changes` | array | Delta from previous step (new events, state changes) |
| `warnings` | array | Deadline alerts and stuck resource notices |
| `last_action_result` | string | Feedback from the previous tool call |
| `last_action_error` | string | Validation error if the last action was illegal |
| `available_tools` | array | The 12 callable tool names |

---

## Action Space

Every action is a JSON tool call:

```json
{"tool": "<tool_name>", "args": {"key": "value"}}
```

| Tool | Arguments | Purpose |
|---|---|---|
| `call_intake_agent` | `report_id` | Full triage: classify + urgency + verification in one call |
| `classify_report` | `report_id` | Classify report category only |
| `assess_report_urgency` | `report_id` | Score urgency (0–10) |
| `verify_report` | `report_id` | Authenticate report; exposes confidence score |
| `mark_false_report` | `report_id, reason` | Flag as false alarm / duplicate |
| `get_resources` | *(none)* | Enumerate full fleet with status |
| `send_resource` | `resource_id, report_id` | Directly dispatch a resource |
| `call_dispatch_agent` | `resource_id, report_id` | Query inventory then dispatch |
| `check_operation` | `target_id` | Poll status of a report or assignment |
| `call_monitor_agent` | `target_id` | Monitor and auto-close if resolved |
| `close_case` | `assignment_id` | Explicitly close a completed assignment |
| `reroute_resource` | `resource_id` | Reroute a stuck or misassigned resource |

---

## Tasks and Difficulty Progression

Three tasks provide a clear difficulty ladder. Complexity increases across every axis simultaneously — more zones, more reports, more false alarms, and stricter grading criteria:

| Task | Difficulty | Steps | Zones | Reports | Resources |
|---|---|---|---|---|---|
| `task1_flood_easy` | Easy | 12 | 1 | 6 | 5 |
| `task2_storm_medium` | Medium | 15 | 3 | 12 | 6 |
| `task3_cascade_hard` | Hard | 20 | 5 | 20 | 6 |

All tasks are generated **deterministically from a seed** — episodes are fully reproducible and fair across evaluation runs.

---

## Grading

Each episode is graded on a **[0.0, 1.0]** scale. The grader evaluates multiple complementary signals rather than a single metric, making it robust against degenerate strategies (e.g., dispatching everything wildly will still fail on resource correctness and false alarm precision).

### Task 1 — Flood Response (Easy)
Focuses on the basics: did the agent resolve incidents and keep critical cases alive?

| Component | What it measures |
|---|---|
| Resolution rate | Fraction of reports successfully resolved |
| Critical resolution | Whether reports marked `is_critical` were resolved before expiry |
| Step efficiency | Penalises excessive tool calls on already-resolved cases |

### Task 2 — Multi-Zone Storm (Medium)
Adds false alarm detection and resource type reasoning:

| Component | What it measures |
|---|---|
| Resolution rate | Overall incident coverage |
| Critical resolution | Critical report survival |
| False alarm F1 | Precision × recall for correctly identified false alarms |
| Resource type match | Whether the dispatched resource type was appropriate for the zone and injury |
| Counterfactual penalty | Deducted when a critical report expired but a valid resource was idle |

### Task 3 — Cascade Disaster (Hard)
Full scoring suite — every grading dimension is active:

| Component | What it measures |
|---|---|
| Resolution rate | Incident coverage across 5 zones |
| Critical resolution | Hard deadline adherence |
| False alarm F1 | Precision and recall balanced |
| Resource type match | Flood-capability, capacity, and type alignment |
| Monitoring quality | Whether the agent checked and closed cases promptly |
| Counterfactual penalty | Missed opportunities despite available resources |

**Reward shaping during episodes:**
- A **temporal urgency multiplier** (2.0× → 1.0×) rewards early action on critical reports
- Per-step rewards are clipped to `[-1, +1]` and accumulated across the episode
- Final episode score is computed by the grader, independent of accumulated reward

---

## Environment Design

The environment follows clean software boundaries with no shared mutable state between episodes:

```
DisasterReliefEnv
├── reset(task, seed)     → StepResult   # deterministic, no state leakage
├── step(action)          → StepResult   # {observation, reward, done, info}
└── grade()               → GradeResult  # end-of-episode score breakdown
```

- **State management:** `WorldState` encapsulates all simulation state. `advance_time()` executes the simulation tick before each observation is built — resource ETAs decrement, deadlines expire, roads unblock, comms restore.
- **Observation builder:** Decoupled from state; produces a clean dict representation without exposing internal objects.
- **Tool registry:** All 12 tools are registered via a central `tool_registry.py` — adding a new tool requires touching only one file.
- **Reward computation:** `rewards.py` is a pure function over `(WorldState, action_result)` — no side effects, fully unit-testable.

---

## API Endpoints

The environment is served as a stateful HTTP API:

| Method | Path | Body / Response |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | List available task names |
| `POST` | `/reset` | `{"task_name": "task1_flood_easy", "seed": 42}` → `StepResult` |
| `POST` | `/step` | `{"tool": "...", "args": {...}}` → `StepResult` |
| `GET` | `/state` | Internal world state snapshot (debug) |
| `POST` | `/grade` | Final episode grade → `GradeResult` |

---

## Code Structure

```
├── app.py                    # FastAPI server — OpenEnv HTTP API
├── inference.py              # Baseline agent: LLM + heuristic fallback
├── environment.py            # Root shim (re-exports DisasterReliefEnv)
├── openenv.yaml              # OpenEnv spec
├── Dockerfile
├── requirements.txt
└── src/env/
    ├── environment.py        # DisasterReliefEnv — reset / step / grade
    ├── models.py             # Pydantic v2 models: 7 enums, 4 domain types, API schemas
    ├── state.py              # WorldState + advance_time() simulation tick
    ├── scenarios.py          # Deterministic scenario generator (seeded RNG)
    ├── observation.py        # Observation builder (stateless)
    ├── rewards.py            # Per-step reward function (pure)
    ├── graders.py            # Episode graders: F1, counterfactual, resource match
    ├── tool_registry.py      # Tool name → handler dispatch table
    ├── tools_intake.py       # classify / assess_urgency / verify_report
    ├── tools_dispatch.py     # get_resources / send_resource / reroute
    ├── tools_monitor.py      # check_operation / close_case / mark_false_report
    └── tools_coordinator.py  # call_intake / dispatch / monitor agent wrappers
```

All domain models are fully typed with Pydantic v2. The test suite covers unit logic, grader bounds, server endpoints, scenario determinism, stdout compliance, and LLM integration smoke tests.

---

## Running Locally

**Requirements:** Python 3.11+, or Docker.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --host 0.0.0.0 --port 7860
```

**Run the baseline agent (heuristic only — no API key needed):**
```bash
python inference.py --heuristic-only
```

**Run with an LLM:**
```bash
export API_BASE_URL="https://router.huggingface.co/together/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-Turbo"
export HF_TOKEN="your-token-here"
python inference.py
```

**Docker:**
```bash
docker build -t disaster-relief .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/together/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-Turbo" \
  -e HF_TOKEN="your-token-here" \
  disaster-relief
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes (default set) | OpenAI-compatible endpoint |
| `MODEL_NAME` | Yes (default set) | Model identifier |
| `HF_TOKEN` | Yes | Bearer token — **no default, must be injected** |
| `PORT` | No | Server port (default: 7860) |

---

## Baseline Scores

Scores from the deterministic flood-aware heuristic agent (seed=42, no LLM required). Reproduce with `python inference.py --heuristic-only`.

| Task | Score | Total Reward | Resolved | Critical Missed |
|---|---|---|---|---|
| `task1_flood_easy` | **0.51** | 7.0 | 2 / 6 | 0 |
| `task2_storm_medium` | **0.18** | 4.0 | 2 / 13 | 0 |
| `task3_cascade_hard` | **0.22** | 1.4 | 3 / 23 | 0 |

The heuristic deliberately scores low on harder tasks to leave meaningful headroom for LLM-based agents to demonstrate reasoning capability — particularly on false alarm F1, resource matching, and counterfactual avoidance.
