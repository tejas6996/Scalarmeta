---
title: Village Microgrid Energy Management
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# Village Microgrid Energy Management — OpenEnv

An **OpenEnv-compatible** reinforcement learning environment where an AI agent
acts as the energy controller for a small village microgrid.

---

## Environment Description

The agent must balance three energy sources and two demand types across a
24-hour (or 72-hour) simulation where **every step = 1 hour**.

**Sources**
| Source | Description |
|---|---|
| Solar panels | Variable generation based on time-of-day profile |
| Battery | 40 kWh capacity, ±10 kWh/step charge/discharge rate |
| Main grid | Unlimited import, but incurs a reward penalty |

**Demand**
| Demand type | Description |
|---|---|
| `critical_demand_kwh` | Hospital + water pump — must NEVER be unmet |
| `residential_demand_kwh` | Homes — can be shed if power is low |

**Energy balance per step (simple arithmetic):**
```
available = solar + grid_import + battery_discharge
used      = critical_demand + residential_supplied + battery_charge
surplus   = available − used   (spilled if positive)
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `step_hour` | int [0,71] | Current simulation hour |
| `solar_gen_kwh` | float ≥ 0 | Solar available this hour |
| `battery_soc_kwh` | float ≥ 0 | Battery state-of-charge |
| `critical_demand_kwh` | float ≥ 0 | Must-serve demand |
| `residential_demand_kwh` | float ≥ 0 | Sheddable demand |

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `grid_import_kwh` | float ≥ 0 | Buy from main grid |
| `battery_action_kwh` | float | + = charge, − = discharge |
| `residential_supplied_kwh` | float ≥ 0 | How much residential to serve |

---

## Reward Function

| Event | Reward |
|---|---|
| Base survival | **+1.0** per step |
| Unmet critical demand | **−5.0** × kWh unmet |
| Unmet residential demand | **−0.5** × kWh unmet |
| Grid import | **−0.1** × kWh imported |
| Invalid action | **−1.0** per violation |

---

## Tasks

| Task | Difficulty | Steps | Description |
|---|---|---|---|
| `task1_summer_day` | Easy | 24 | High solar, low demand, battery starts empty |
| `task2_winter_night` | Medium | 24 | Minimal solar, battery starts full, grid is expensive |
| `task3_rolling_blackout` | Hard | 72 | Erratic weather, 3 kWh/step grid import cap |

### Grading

| Task | Score Logic |
|---|---|
| Task 1 | 0.0 if any critical failure; else 50% battery end-level + 50% residential service |
| Task 2 | 0.0 if any critical failure; else penalises total grid import |
| Task 3 | 60% critical uptime fraction + 40% residential service fraction |

All scores are in **[0.0, 1.0]**.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | List supported tasks |
| `POST` | `/reset` | Start new episode `{"task_name": "..."}` |
| `POST` | `/step` | Take action `{"action": {...}}` |
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

### Run the baseline inference script
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-token-here"
python inference.py
```

### Run environment sanity check
```bash
python environment.py
```

### Docker build & run
```bash
docker build -t village-microgrid .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your-token-here" \
  village-microgrid
```

---

## Project Structure

```
village-microgrid-env/
├── environment.py     # Core Pydantic models, VillageMicrogridEnv, Graders
├── app.py             # FastAPI HTTP server (OpenEnv API)
├── inference.py       # Baseline LLM inference script (required by spec)
├── openenv.yaml       # OpenEnv spec file
├── requirements.txt
├── Dockerfile
└── README.md
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

## Baseline Scores

Scores produced by the deterministic greedy heuristic (no LLM required).
Run `python inference.py` with `HF_TOKEN` unset to reproduce exactly.

| Task | Score | Total Reward | Critical Failures | Notes |
|---|---|---|---|---|
| `task1_summer_day` | **0.5000** | 17.21 | 0 | All demand met; battery did not reach 90% by end of day |
| `task2_winter_night` | **0.5020** | 14.04 | 0 | No blackouts; score penalised for grid imports |
| `task3_rolling_blackout` | **0.8429** | −10.60 | 0 | 72 h with erratic solar; near-perfect critical uptime |

Scores are deterministic — the heuristic is seeded and the solar profiles are fixed.
Achieve higher scores by improving the LLM prompt or implementing a planning algorithm.

---

## Motivation

Energy management for isolated village microgrids is a genuinely hard real-time
optimisation problem faced by millions of communities in rural India, sub-Saharan
Africa, and Southeast Asia. Decisions made by the grid controller directly affect
whether a hospital stays powered, whether water pumps run, and whether homeowners
have light after sunset — all under strict hardware limits and unpredictable solar
generation.

This environment makes that decision-making task learnable by an AI agent.
It faithfully models the key trade-offs (battery degradation vs. grid cost vs.
load shedding) at a level of fidelity useful for benchmarking planning and RL
algorithms without requiring hardware-in-the-loop simulation.
