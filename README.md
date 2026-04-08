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

# Disaster Relief Coordination — OpenEnv

An **OpenEnv-compatible** environment where an LLM coordinator triages
disaster reports, allocates limited resources, and resolves incidents
under time pressure, uncertainty, and flood conditions.

---

## Environment Description

The agent acts as a **disaster relief coordination AI** managing a
multi-zone disaster area. Each step, the agent must decide which tool
to invoke — triaging incoming reports, dispatching resources, flagging
false alarms, or monitoring active operations.

**Key Challenges**
| Challenge | Description |
|---|---|
| Flood zones | Zones with flood depth ≥ 2 require flood-capable resources (boats, helicopters) |
| False/duplicate reports | Agent must identify and flag false alarms without penalizing real incidents |
| Critical deadlines | Critical reports expire if not resolved in time |
| Resource constraints | Limited resources with different types, fuel, and flood capability |
| Comms blackouts | Some zones temporarily lose communication |
| Road blockages | Some zones have blocked access that clears mid-episode |

**Resources**
| Type | Flood-capable | Description |
|---|---|---|
| Ambulance | No | Medical transport, capacity 2 |
| Rescue Boat | Yes | Flood rescue, capacity 4 |
| Supply Truck | No | Supplies/evacuation, capacity 20 |
| Medical Team | No | On-site medical care, capacity 3 |
| Helicopter | Yes | All-terrain, capacity 2 |
| Engineering Crew | No | Structural/road repair, capacity 5 |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `step` | int | Current step (0-based) |
| `max_steps` | int | Total steps in this episode |
| `pending_reports` | array | Visible reports not yet resolved |
| `active_assignments` | array | Dispatched resource assignments |
| `available_resources` | array | All resources with type, status, flood capability |
| `zones` | array | Zone conditions: flood depth, access, comms |
| `recent_changes` | array | What changed since last step |
| `warnings` | array | Deadline warnings, stuck resources |
| `available_tools` | array | 12 tool names the agent can call |
| `last_action_result` | string | Feedback from previous tool call |
| `last_action_error` | string | Error if previous action was invalid |

---

## Action Space

Each action is a tool call:
```json
{"tool": "<tool_name>", "args": {<arguments>}}
```

**Available Tools (12)**
| Tool | Arguments | Description |
|---|---|---|
| `call_intake_agent` | `report_id` | Classify + assess urgency + verify a report |
| `send_resource` | `resource_id, report_id` | Dispatch a resource to a report |
| `call_dispatch_agent` | `resource_id, report_id` | Query inventory then dispatch |
| `mark_false_report` | `report_id, reason` | Flag a report as false alarm |
| `check_operation` | `target_id` | Check status of report or assignment |
| `call_monitor_agent` | `target_id` | Monitor and optionally close a case |
| `close_case` | `assignment_id` | Close a completed assignment |
| `get_resources` | *(none)* | List all resources and status |
| `classify_report` | `report_id` | Classify a single report |
| `assess_report_urgency` | `report_id` | Score a report's urgency |
| `verify_report` | `report_id` | Verify a report's authenticity |
| `reroute_resource` | `resource_id` | Reroute a stuck resource |

---

## Tasks

| Task | Difficulty | Steps | Zones | Reports | Resources |
|---|---|---|---|---|---|
| `task1_flood_easy` | Easy | 12 | 1 | 6 | 5 |
| `task2_storm_medium` | Medium | 15 | 3 | 12 | 6 |
| `task3_cascade_hard` | Hard | 20 | 5 | 20 | 6 |

### Grading

| Task | Score Formula |
|---|---|
| Task 1 | 40% resolution + 30% critical + 30% efficiency |
| Task 2 | 25% resolution + 20% critical + 20% F1 + 20% resource match + 15% counterfactual |
| Task 3 | 20% resolution + 20% critical + 20% F1 + 15% resource + 15% monitoring + 10% counterfactual |

**Advanced grading features:**
- **F1-score** for false alarm detection (precision × recall)
- **Resource type correctness** — did the agent send the right resource?
- **Counterfactual penalty** — was a correct resource available when a critical report expired?
- **Temporal urgency multiplier** — rewards scale 2.0x→1.0x as deadlines approach

All scores are in **[0.0, 1.0]**.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | List supported tasks |
| `POST` | `/reset` | Start new episode `{"task_name": "...", "seed": 42}` |
| `POST` | `/step` | Take action `{"tool": "...", "args": {...}}` |
| `GET` | `/state` | Internal state snapshot |
| `POST` | `/grade` | Grade completed episode |

---

## Setup & Running Locally

### Prerequisites
- Python 3.11+
- Docker (optional, for containerised run)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run inference (heuristic-only, no API key needed)
```bash
python inference.py --heuristic-only
```

### Run inference (with LLM)
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-token-here"
python inference.py
```

### Docker build & run
```bash
docker build -t disaster-relief .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your-token-here" \
  disaster-relief
```

---

## Project Structure

```
disaster-relief-coordination/
├── app.py               # FastAPI HTTP server (OpenEnv API)
├── inference.py          # LLM inference + heuristic fallback
├── environment.py        # Root shim → src/env/environment.py
├── openenv.yaml          # OpenEnv spec file
├── requirements.txt
├── Dockerfile
├── README.md
├── server/
│   └── app.py            # Uvicorn entry point
└── src/env/
    ├── models.py          # All Pydantic models (7 enums, 4 domain, API schemas)
    ├── scenarios.py       # Deterministic scenario generator (3 tasks)
    ├── state.py           # WorldState with advance_time() simulation tick
    ├── tools_intake.py    # classify, urgency, verify
    ├── tools_dispatch.py  # get_resources, send_resource, reroute
    ├── tools_monitor.py   # check_operation, close_case, mark_false
    ├── tools_coordinator.py # call_intake/dispatch/monitor_agent
    ├── tool_registry.py   # Tool name → handler registry
    ├── rewards.py         # Per-step reward computation
    ├── graders.py         # Episode grading (F1, counterfactual, etc.)
    ├── observation.py     # Observation builder
    └── environment.py     # DisasterReliefEnv class
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `API_BASE_URL` | Base URL of the OpenAI-compatible LLM API |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Hugging Face / API key used as bearer token |
| `PORT` | Server port (default: 7860) |

Set these as **Repository Secrets** in your Hugging Face Space.

---

## Baseline Scores (Heuristic)

Scores produced by the deterministic flood-aware heuristic (no LLM required).
Run `python inference.py --heuristic-only` to reproduce.

| Task | Score | Total Reward | Resolved | Critical Missed |
|---|---|---|---|---|
| `task1_flood_easy` | **0.5100** | 7.0 | 2/6 | 0 |
| `task2_storm_medium` | **0.1822** | 4.0 | 2/13 | 0 |
| `task3_cascade_hard` | **0.2221** | 1.4 | 3/23 | 0 |

Scores are deterministic (seed=42). Achieve higher scores by improving LLM prompting
or implementing a planning algorithm.
