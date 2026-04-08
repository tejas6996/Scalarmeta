# Build Plan — Disaster Relief Coordination OpenEnv

## Current State
- Repo contains a **working Village Microgrid environment** (energy management).
- We are on the `agent` branch. The entire environment concept must be **replaced** with the Disaster Relief Coordination Environment.
- Existing scaffolding to **keep and adapt**: `Dockerfile`, `validate-submission.ps1`, `validate-submission.sh`, `server/`, `pyproject.toml`, `requirements.txt`.
- Everything in `environment.py`, `app.py`, `inference.py`, `openenv.yaml`, `README.md` must be **rewritten from scratch**.
- `src/env/` exists but is empty — this is where the new environment package will live.

---

## Phase 1: Core Data Models
**Goal:** Define every Pydantic model the system needs before writing any logic.

### Steps
1. **`src/env/models.py`** — Define all typed models:
   - `Report` (id, raw_text, zone, category, urgency, ground_truth_real, ground_truth_duplicate_of, required_resource_type, deadline_step, status, created_step, resolved_step, assigned_resource_id)
   - `Resource` (id, type, status: available/deployed/returning, assigned_report_id, location, eta_available_step)
   - `Zone` (id, name, severity, access_blocked, open_incidents)
   - `Assignment` (id, resource_id, report_id, created_step, route_status, status, expected_completion_step, stuck)
   - `Observation` — the dict/model the LLM sees each step (step number, max_steps, pending_reports summary, active_assignments summary, available_resources summary, recent_changes, available_tools list, warnings)
   - `Action` — single tool call: `{"tool": str, "args": dict}`
   - `StepResult` — (observation, reward, done, info)
   - `Reward` — structured breakdown (base, triage, dispatch, monitor, penalty, total)
   - `EnvironmentState` — public snapshot for `/state`
   - `EpisodeMemoryEntry` — (step, actor, tool, args, summary, result_status, reward, entities)

2. **`src/env/__init__.py`** — Export public API.

**Done when:** All models importable, `python -c "from src.env.models import *"` succeeds.

---

## Phase 2: Scenario Generator
**Goal:** Deterministically generate reports, resources, zones, and ground truth for each task.

### Steps
1. **`src/env/scenarios.py`** — Implement `generate_scenario(task_name, seed)` returning:
   - List of `Report` objects (with staggered `created_step` so they arrive over time)
   - List of `Resource` objects (ambulances, rescue boats, supply trucks, medical teams, etc.)
   - List of `Zone` objects (with varying severity/access)
   - Hidden ground truth dict (which reports are real, which are duplicates, which are false)
   - Task parameters (max_steps, report_count, duplicate_rate, false_rate, blockage_rate)

2. Define **3 task configs**:
   - `task1_flood_easy` — 1 zone, ~6 reports (1 false, 0 duplicates), 5 resources, 12 steps, no road blocks
   - `task2_storm_medium` — 3 zones, ~12 reports (2 false, 2 duplicates), 6 resources, 18 steps, 1-2 blocked routes
   - `task3_cascade_hard` — 5 zones, ~20 reports (3 false, 4 duplicates), 6 resources, 25 steps, multiple blockages, tight deadlines

3. Generate realistic synthetic report texts using templates + seeded randomization (e.g., "Flooding at {location}, {n} people trapped, need {resource_type}").

**Done when:** `generate_scenario("task1_flood_easy", seed=42)` returns a fully populated, deterministic scenario.

---

## Phase 3: World State & State Transitions
**Goal:** Build the mutable state container that all tools read from and write to.

### Steps
1. **`src/env/state.py`** — `WorldState` class:
   - Holds reports (dict by id), resources (dict by id), zones (dict by id), assignments (dict by id)
   - Holds episode memory list, reward accumulators, step counter
   - Methods: `get_pending_reports(current_step)`, `get_available_resources()`, `get_active_assignments()`, `advance_time()` (move ETAs, mark completions, surface new reports)
   - `advance_time()` is the key dynamic method — called each step to:
     - Release reports whose `created_step <= current_step`
     - Progress assignments (decrement ETAs, mark completed if done)
     - Detect stuck/delayed assignments
     - Update zone incident counts

**Done when:** State can be initialized from scenario output, and `advance_time()` correctly progresses the world.

---

## Phase 4: Tool Implementations
**Goal:** Build all 12 tools as pure functions that take state + args and return (result_text, state_mutations, reward_delta).

### Steps
1. **`src/env/tools_intake.py`** — 3 intake tools:
   - `classify_report(state, report_id)` → category, location, type extraction
   - `assess_report_urgency(state, report_id)` → urgency score + reasoning
   - `verify_report(state, report_id)` → real/duplicate/false confidence

2. **`src/env/tools_dispatch.py`** — 3 dispatch tools:
   - `get_resources(state)` → available + deployed inventory
   - `send_resource(state, resource_id, report_id)` → attempt dispatch, create Assignment, update Resource
   - `reroute_resource(state, resource_id, route_hint)` → handle blocked routes

3. **`src/env/tools_monitor.py`** — 3 monitor tools:
   - `check_operation(state, target_id)` → assignment or report status
   - `close_case(state, report_id, resolution_note)` → mark resolved
   - `mark_false_report(state, report_id, reason)` → flag as false/duplicate

4. **`src/env/tools_coordinator.py`** — 3 delegation tools:
   - `call_intake_agent(state, report_id, instruction)` → chains intake tools internally, returns specialist brief
   - `call_dispatch_agent(state, resource_id, report_id)` → chains dispatch tools internally
   - `call_monitor_agent(state, target_id, instruction)` → chains monitor tools internally

5. **`src/env/tool_registry.py`** — Registry mapping tool name strings → handler functions. Validates args.

**Done when:** Each tool can be called with valid state+args and returns a correct state-derived response. Invalid IDs return structured errors, not crashes.

---

## Phase 5: Reward Function
**Goal:** Dense per-step reward that reflects the masterplan's reward design.

### Steps
1. **`src/env/rewards.py`** — `compute_step_reward(state, tool_name, tool_args, tool_result, step)`:
   - `+0.5` base survival
   - `+1.0` correct triage of a new report
   - `+2.0` correct dispatch (right resource type to real report)
   - `+1.5` correct case closure
   - `+1.0` correctly identifying false/duplicate
   - `-1.0` dispatching to false/duplicate report
   - `-1.0` wrong resource type
   - `-2.0` missed critical deadline
   - `-0.5` repeated identical tool call
   - `-0.5` malformed action
   - `-0.5` premature closure
   - `-1.5` marking a real report as false

**Done when:** Reward function produces different values for good vs bad actions in unit tests.

---

## Phase 6: Graders
**Goal:** Episode-end scoring in [0.0, 1.0] per task.

### Steps
1. **`src/env/graders.py`** — 3 grader classes:
   - `Task1Grader` — 40% resolution completeness + 30% critical handling + 30% efficiency
   - `Task2Grader` — 30% resolution + 25% critical + 20% verification accuracy + 25% resource correctness
   - `Task3Grader` — 25% resolution + 25% critical + 20% verification + 15% resource correctness + 15% monitoring

2. Each grader reads `state.info_history` + hidden ground truth and scores deterministically.

**Done when:** Same seed + same actions = same score. Graders return 0.0 for do-nothing episodes and >0.5 for reasonable play.

---

## Phase 7: Observation Builder & Memory
**Goal:** Build the compressed observation the LLM sees each step.

### Steps
1. **`src/env/observation.py`** — `build_observation(state)`:
   - Current step / max steps
   - Pending unresolved reports (brief summaries, capped to most recent/urgent)
   - Active assignments snapshot
   - Available resources count by type
   - Recent changes since last step (new reports, completions, failures)
   - Warnings (approaching deadlines, stuck assignments, empty inventory)
   - Available tools list with short signatures

2. **`src/env/memory.py`** — Episode memory:
   - Append structured entry each step
   - `get_rolling_summary(last_n=5)` for observation compression

**Done when:** Observation is a clean JSON-serializable dict, compact enough for LLM context windows.

---

## Phase 8: Main Environment Class
**Goal:** Wire everything together into `DisasterReliefEnv` with `reset()`, `step()`, `state()`, `grade_episode()`.

### Steps
1. **`src/env/environment.py`** (or rename to `environment.py` at root):
   - `reset(task_name)` → generate scenario, init state, return first observation
   - `step(action)` → parse tool call, route to handler, update state, compute reward, advance time, build next observation, return (obs, reward, done, info)
   - `state()` → return public snapshot
   - `grade_episode()` → delegate to task-specific grader
   - Handle malformed actions gracefully (error observation + small penalty, episode continues)

2. Move/replace root `environment.py` to import from `src/env/`.

**Done when:** Full episode loop works: `reset → step × N → grade` with deterministic output.

---

## Phase 9: FastAPI Server (`app.py`)
**Goal:** Expose the environment via HTTP endpoints matching OpenEnv spec.

### Steps
1. Rewrite root **`app.py`**:
   - `GET /` → health check
   - `GET /tasks` → list task names
   - `POST /reset` → `{"task_name": "..."}` → initial observation
   - `POST /step` → `{"action": {"tool": "...", "args": {...}}}` → (obs, reward, done, info)
   - `GET /state` → environment state snapshot
   - `POST /grade` → episode score

**Done when:** `uvicorn app:app` starts, all endpoints return correct JSON.

---

## Phase 10: Inference Script (`inference.py`)
**Goal:** Baseline LLM-driven coordinator that runs all 3 tasks end-to-end.

### Steps
1. Rewrite root **`inference.py`**:
   - Read `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from env vars
   - OpenAI client setup
   - System prompt for coordinator role (tool signatures, strategy guidance, one-action-per-turn)
   - Per-step loop: send observation → parse LLM tool call → send to env → log
   - Fallback heuristic if LLM output unparseable
   - **Exact logging format**:
     - `[START] task=<name> env=disaster-relief model=<model>`
     - `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
     - `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`
   - Always emit `[END]` even on exception

2. Heuristic fallback: simple priority-based loop (intake highest urgency → dispatch first available matching resource → monitor oldest assignment → close if done).

**Done when:** `python inference.py` runs all 3 tasks, emits correct logs, finishes in <20 min.

---

## Phase 11: OpenEnv Spec & Config Files
**Goal:** Update all config files for the new environment.

### Steps
1. Rewrite **`openenv.yaml`** — observation/action schemas, 3 tasks, reward description, endpoints, hardware constraints.
2. Rewrite **`README.md`** — full environment description, action/observation space, tasks, rewards, graders, setup, env vars, validation.
3. Update **`requirements.txt`** — ensure all deps listed.
4. Update **`Dockerfile`** — should mostly work as-is, verify CMD.
5. Update **`pyproject.toml`** if needed.

**Done when:** `openenv.yaml` matches the actual implementation.

---

## Phase 12: Testing & Validation
**Goal:** Ensure everything passes before submission.

### Steps
1. **Unit tests** (can be lightweight scripts):
   - Scenario determinism: same seed → same scenario
   - Each tool returns valid output for valid input and structured error for invalid input
   - Reward values match expected for known good/bad actions
   - Grader scores: do-nothing ≈ 0.0, perfect play ≈ 1.0
   - State transitions are consistent

2. **Integration tests**:
   - Full episode loop (reset → N steps → grade) for each task
   - Malformed action handling
   - Double-close, invalid IDs, zero resources edge cases

3. **Validator flow**:
   - `docker build -t disaster-relief .`
   - `docker run -p 7860:7860 disaster-relief`
   - Ping `GET /` → 200
   - `POST /reset` → valid observation
   - Run `validate-submission.ps1`
   - Run `python inference.py` end-to-end

**Done when:** All validation checks pass, Docker builds clean, inference completes <20 min.

---

## Phase 13: Deploy to HF Space
**Goal:** Live on Hugging Face Spaces.

### Steps
1. Push to HF Space repo
2. Set secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
3. Verify Space builds and starts
4. Verify `/reset` returns 200
5. Run remote inference test

---

## Dependency Graph

```
Phase 1 (Models)
    ↓
Phase 2 (Scenarios)  ←──── Phase 3 (State)
    ↓                         ↓
Phase 4 (Tools) ──────────────┘
    ↓
Phase 5 (Rewards)
    ↓
Phase 6 (Graders)
    ↓
Phase 7 (Observation + Memory)
    ↓
Phase 8 (Environment Class) ← wires Phases 2-7
    ↓
Phase 9 (FastAPI)  ←── Phase 10 (Inference)
    ↓                      ↓
Phase 11 (Config Files)
    ↓
Phase 12 (Testing)
    ↓
Phase 13 (Deploy)
```

---

## File Map (Final)

```
Scalarmeta/
├── app.py                      # FastAPI server (Phase 9)
├── environment.py              # Thin re-export from src/env (Phase 8)
├── inference.py                # Baseline LLM script (Phase 10)
├── openenv.yaml                # OpenEnv spec (Phase 11)
├── README.md                   # Documentation (Phase 11)
├── Dockerfile                  # Docker build (Phase 11)
├── requirements.txt            # Dependencies (Phase 11)
├── pyproject.toml
├── validate-submission.ps1
├── validate-submission.sh
├── docs/
│   ├── masterplan.md
│   └── buildplan.md
├── server/
│   ├── __init__.py
│   └── app.py
├── src/
│   └── env/
│       ├── __init__.py         # Public exports (Phase 1)
│       ├── models.py           # All Pydantic models (Phase 1)
│       ├── scenarios.py        # Deterministic scenario gen (Phase 2)
│       ├── state.py            # WorldState class (Phase 3)
│       ├── tools_intake.py     # Intake tools (Phase 4)
│       ├── tools_dispatch.py   # Dispatch tools (Phase 4)
│       ├── tools_monitor.py    # Monitor tools (Phase 4)
│       ├── tools_coordinator.py# Coordinator delegation tools (Phase 4)
│       ├── tool_registry.py    # Tool name → handler map (Phase 4)
│       ├── rewards.py          # Step reward function (Phase 5)
│       ├── graders.py          # Task graders (Phase 6)
│       ├── observation.py      # Observation builder (Phase 7)
│       ├── memory.py           # Episode memory (Phase 7)
│       └── environment.py      # DisasterReliefEnv class (Phase 8)
└── tests/                      # Optional test scripts (Phase 12)
```
