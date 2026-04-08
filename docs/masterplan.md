# Master Plan — Disaster Relief Coordination OpenEnv

## 1. Project Identity

**Project name:** Disaster Relief Coordination Environment  
**Round:** Meta / PyTorch / Hugging Face OpenEnv Hackathon — Round 1  
**Category:** Real-world workflow environment for RL-ready LLM interaction  
**Primary goal:** Build a complete OpenEnv-compatible environment where an LLM coordinator learns to triage disaster reports, allocate limited resources, monitor active operations, and resolve incidents under uncertainty.

This project is intentionally designed for **Round 1**, which is about building the **environment**, not building a production agent. The environment must still support a real baseline `inference.py` that calls an LLM through the OpenAI client and runs end-to-end without errors.

---

## 2. Hackathon Requirements — What Must Be True

The attached Round 1 brief establishes the following hard requirements:

- The environment must simulate a **real-world task**, not a toy or game.
- It must implement the full OpenEnv spec, including typed models and `reset()`, `step()`, and `state()`.
- It must include **at least 3 tasks** with increasing difficulty and graders that return scores in `[0.0, 1.0]`.
- It must use a **meaningful reward function** with partial progress signals.
- It must include a reproducible `inference.py` baseline.
- It must deploy to **Hugging Face Spaces** and build successfully via Docker.
- It must expose the environment in a way that passes automated validation.
- It must finish inference within **20 minutes** on **2 vCPU / 8 GB RAM**.
- The validator will ping `/reset`, run Docker build, and run `openenv validate`.
- The baseline script must use the **OpenAI client** and read `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from env vars.
- The baseline script must emit exact structured stdout lines: `[START]`, `[STEP]`, `[END]` in the required format.

These are not suggestions. Failing any of them can disqualify the submission.

---

## 3. Final Environment Concept

### Core simulation
A district-level disaster response coordination center receives noisy incoming reports during a flood / storm / cascading emergency. The LLM plays the role of a **coordinator**. It must:

1. Understand incoming reports.
2. Decide which report matters most right now.
3. Delegate specialist analysis when needed.
4. Dispatch the correct limited resource.
5. Monitor active assignments.
6. Close resolved situations.
7. Avoid wasting resources on duplicates, false reports, or low-priority distractions.

### Why this concept is strong
- It is unmistakably a **real-world workflow**.
- It is text-native and ideal for LLM interaction.
- It supports incremental reward and partial progress.
- It naturally supports multi-step reasoning, prioritization, and uncertainty.
- It is easy to score because ground truth is seeded at episode creation.
- It is rich enough to impress judges without requiring external APIs beyond the LLM baseline.

---

## 4. Strict Non-Mock Rule

### Policy
This project must use **no mock / placeholder / fake hardcoded outputs** in the behavior layer.

### What is allowed
- Procedurally generated scenarios.
- Seeded synthetic reports.
- Deterministic simulation logic.
- Dynamic text responses generated from state.
- Synthetic but internally consistent resources, ETAs, road blocks, deadlines, and report streams.

### What is forbidden
- Returning static canned strings unrelated to episode state.
- Returning the same success response for every tool call.
- Pretending a resource moved or a report was resolved without updating state.
- Writing graders that ignore actual episode behavior.
- Using toy placeholder values such as `"done"`, `"ok"`, `"resource assigned"` without context.

### Engineering interpretation
The data may be synthetic, but the **logic must be real**. Every observation, tool result, reward, and grader score must be derived from the current state and hidden ground truth.

---

## 5. Roles in the System

We will use a **lean 3-specialist + 1 coordinator design**.

### 5.1 Coordinator
This is the main LLM used in `inference.py`. It is the only external agent loop. It reads the observation and decides the next action.

Responsibilities:
- Decide what information is missing.
- Decide whether to delegate to a specialist.
- Decide whether to dispatch immediately.
- Track unresolved work.
- Keep the episode moving and avoid idle loops.

### 5.2 Intake Agent (internal specialist)
Purpose: make sense of raw reports.

Responsibilities:
- Classify report type.
- Extract location.
- Identify duplicate risk.
- Estimate urgency.
- Verify whether the report is likely real, duplicate, or false.

### 5.3 Dispatch Agent (internal specialist)
Purpose: handle resource allocation.

Responsibilities:
- Match resource type to need.
- Check availability.
- Deploy resource.
- Recompute routes when blocked.
- Surface routing or capacity constraints.

### 5.4 Monitor Agent (internal specialist)
Purpose: track active work and keep the queue healthy.

Responsibilities:
- Check assignment status.
- Identify delays or failures.
- Confirm completion.
- Mark incidents resolved.
- Flag false / duplicate reports when discovered later.

### Important implementation truth
There is still only **one outer inference loop**. The other agents are internal specialist modules invoked through environment tools. This keeps the system fully compatible with OpenEnv while still giving the environment multi-agent structure.

---

## 6. Final Tool Set

We reduce the design to **12 tools total**: 3 per specialist domain + 3 coordinator delegation tools.

## 6.1 Coordinator Delegation Tools
These are the main tools the coordinator will most often call.

### `call_intake_agent(report_id, instruction)`
Use when the coordinator needs report understanding.

Returns:
- Structured specialist brief containing category, extracted location, urgency estimate, duplicate suspicion, verification status, and recommendation.

Behavior:
- Internally invokes Intake logic.
- May chain `classify_report`, `assess_report_urgency`, and `verify_report` internally.
- Writes the result into episodic memory.

### `call_dispatch_agent(resource_id, report_id)`
Use when the coordinator wants to deploy help.

Returns:
- Specialist dispatch decision containing whether the resource is valid, route result, ETA, capacity warning, and assignment ID if successful.

Behavior:
- Internally invokes Dispatch logic.
- May chain `get_resources`, `send_resource`, and `reroute_resource` internally.
- Mutates the state if dispatch succeeds.

### `call_monitor_agent(target_id, instruction)`
Use when the coordinator needs operational follow-up.

Returns:
- Specialist status review, such as assignment state, delay risk, resolution status, or false-report confirmation.

Behavior:
- Internally invokes Monitor logic.
- May chain `check_operation`, `close_case`, or `mark_false_report` internally.
- Updates memory and state.

## 6.2 Intake Tools

### `classify_report(report_id)`
Returns a structured interpretation of the raw report.

### `assess_report_urgency(report_id)`
Returns urgency judgment.

### `verify_report(report_id)`
Returns confidence that the report is real.

## 6.3 Dispatch Tools

### `get_resources()`
Returns currently available and deployed resources.

### `send_resource(resource_id, report_id)`
Attempts deployment.

### `reroute_resource(resource_id, route_hint)`
Handles blocked routes or rerouting requests.

## 6.4 Monitor Tools

### `check_operation(target_id)`
Checks assignment or case progress.

### `close_case(report_id, resolution_note)`
Closes a resolved report.

### `mark_false_report(report_id, reason)`
Flags a case as false or duplicate.

---

## 7. Tool Naming Philosophy

Tool names must be short, readable, unambiguous, parseable, and human understandable under pressure.

---

## 8. Tool Return Rules

Every tool response must be state-derived and must contain enough information to help the coordinator decide the next step.

Required properties:
- reflects current state
- mentions usable identifiers
- concise enough for repeated LLM consumption
- failure reasons when relevant
- visible consequences when a bad decision is made

---

## 9. Memory Design — Episode Working Memory

All LLM-facing interactions are tracked in an **episode memory array**. Each step appends a structured record.

Memory categories:
- action history
- tool call history
- specialist briefs
- unresolved priority snapshots
- active assignment snapshots
- mistakes / warnings
- resolution history

Each memory item should include:
- `step_number`
- `actor`
- `tool_name`
- `input_args`
- `summary`
- `result_status`
- `reward`
- `important_entities`

Policy:
- keep full memory internally
- expose compressed rolling summary to the coordinator each step
- always include recent changes, unresolved criticals, and active assignments

---

## 10. World State Design

The environment state must be the authoritative source of truth.

State domains:
- episode metadata
- reports
- resources
- zones
- assignments
- specialist outputs
- memory history
- hidden grader truth
- reward counters
- loop-prevention markers

Reports track:
- raw text
- created step
- report class
- urgency
- real / duplicate / false status
- deadline
- linked original if duplicate
- required resource type
- resolution state

Resources track:
- id
- type
- current status
- assigned case
- location
- eta / next availability

Assignments track:
- assignment id
- resource id
- report id
- created step
- route status
- current status
- expected completion window
- stuck / delayed flags

Zones track:
- severity
- access constraints
- open incident count
- congestion or blockage

---

## 11. Dynamic Scenario Generation

Goals:
- deterministic with seed
- realistic enough for judging
- different across episodes
- scalable across 3 tasks

Easy:
- single-zone
- fewer reports
- low duplication
- minimal false reports
- simple routing

Medium:
- multi-zone
- more reports
- duplicates and some false reports
- limited resources
- blocked routes
- delayed assignments possible

Hard:
- cascading disaster
- highest report count
- high duplicate and verification noise
- multiple critical deadlines
- assignment failures
- rerouting pressure

Variation knobs:
- report count
- critical count
- duplicate rate
- false report rate
- scarcity
- blockage rate
- deadline tightness

---

## 12. Flow Navigation — Preventing Deadlocks

The observation must always include:
- current step and max steps
- top unresolved reports
- active assignments
- available resources snapshot
- what changed since last step
- available tools reminder
- any hard warning: deadline risk, stuck assignment, empty inventory, duplicate suspicion

The environment should detect repetitive behavior and surface anti-loop feedback.
If the agent makes a malformed action, return a valid error observation, apply a small penalty, and keep the episode alive.

---

## 13. Reward Function Design

Reward must be dense enough to teach but not exploitable.

Positive signals:
- reading before acting
- correct prioritization
- correct specialist delegation
- correct dispatch
- timely monitoring
- correct closure
- correct duplicate / false handling

Negative signals:
- repeated tool calls with no new value
- wrong resource type
- dispatching to false / duplicate report
- premature closure
- marking real report as false
- ignoring overdue critical case
- malformed action output

Reward must depend on:
- action parameters
- current step
- report hidden truth
- current resource state
- prior actions
- whether the action improved the world state

Episode-end score must be in `[0.0, 1.0]`.

---

## 14. Grader Design

The grader evaluates the full trajectory, not just the last action.

Dimensions:
1. Resolution completeness
2. Critical handling quality
3. Decision efficiency
4. Resource correctness
5. Verification quality
6. Monitoring quality

The same seed plus same actions must yield the same final score.

---

## 15. Step Function Design

Conceptual pipeline:
1. validate action payload
2. parse tool call from message
3. route to tool / specialist
4. execute state-aware logic
5. update world state
6. update episodic memory
7. compute step reward
8. advance time / deadlines / assignment status
9. rebuild observation
10. determine `done`
11. attach `last_action_error` if needed

---

## 16. Reset Function Design

`reset()` must:
- choose task
- seed scenario deterministically
- build initial state
- initialize report queue
- initialize resources and assignments
- initialize memory array
- initialize hidden grader truth
- return a usable first observation

---

## 17. State Function Design

`state()` should expose public-safe episode status for inspection and debugging without leaking hidden grader truth.

---

## 18. Prompt Design Guide — Coordinator

The coordinator prompt must contain:
- role identity
- mission objective
- exact allowed output format: one tool call only
- exact tool signatures with examples
- strategy guidance: read → delegate → act → monitor → close
- loop avoidance reminder
- one-action-per-turn reminder
- reminders to track report, assignment, and resource IDs

It should not include chain-of-thought requests or ambiguous formatting.

---

## 19. Prompt Design Guide — Intake Agent

Must contain:
- role: report-understanding specialist
- scope: classify, assess urgency, verify
- concise structured output requirement
- instruction to identify missing or contradictory info
- instruction to recommend next best follow-up

---

## 20. Prompt Design Guide — Dispatch Agent

Must contain:
- role: resource allocation specialist
- scope: inventory, suitability, dispatch feasibility, routing
- instruction to prioritize feasibility and correctness
- instruction to surface conflicts, unavailability, and blockage
- instruction to return assignment IDs and ETA-oriented outcomes

---

## 21. Prompt Design Guide — Monitor Agent

Must contain:
- role: operation tracking specialist
- scope: assignment follow-up, closure readiness, false-report handling
- instruction to detect delay, failure, staleness, and unresolved work
- instruction to avoid premature closure
- instruction to tell the coordinator what changed and what needs follow-up next

---

## 22. Orchestration Model

Coordinator is the outer policy. Specialists are internal expert services.

Flow:
1. coordinator sees observation
2. coordinator chooses one tool
3. if tool is a `call_*_agent` tool, environment invokes that specialist
4. specialist may internally use domain tools
5. environment returns specialist brief
6. coordinator continues

---

## 23. Progress Tracking

Track three levels:
- episode-level progress
- operational progress
- policy progress

Useful metrics:
- real incidents resolved
- critical incidents still open
- wasted dispatches
- deadlines missed
- active assignments count
- average assignment age
- repeated actions count
- malformed actions count

---

## 24. Repository Structure

Recommended repo layout:
- `inference.py` at repo root
- `openenv.yaml` at repo root
- `Dockerfile` at repo root
- `README.md` at repo root
- environment package with typed models, env class, scenario generator, reward logic, grader logic, memory logic, observation builder
- optional scripts folder for local validation helpers

---

## 25. Environment Variables

Primary env vars from the brief:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Additional practical vars:
- image name if using docker image bootstrapping
- task selector for local runs
- benchmark / environment naming vars if exposed

Rules:
- `inference.py` must read env vars
- use OpenAI client
- never hardcode secrets

---

## 26. Inference Script Requirements

`inference.py` must:
- live at repo root
- use OpenAI client
- read env vars
- call environment end-to-end
- emit exact stdout log lines
- always print `[END]` even on exception

Required stdout lines:
- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

Keep booleans lowercase and reward formatting exact.

---

## 27. Testing Strategy

Unit tests:
- deterministic scenario generation
- tool return correctness
- state transitions
- reward behavior
- grader stability
- parser robustness

Simulation tests:
- happy path
- duplicate mishandling
- blocked route reroute
- false report
- late monitoring

Validator parity tests:
1. ping `/reset`
2. `docker build`
3. `openenv validate`
4. run baseline inference end-to-end

Edge cases:
- malformed action
- invalid ids
- double close
- already deployed resource
- zero available resources

---

## 28. Deployment and Exposure

The HF Space must be live and return HTTP 200 on `/reset`.

Checklist:
- server runs on HF Space
- endpoints live
- Docker builds
- no local-only assumptions
- no missing secrets at startup
- startup and baseline runtime stay within constraints

Strong recommendation for Round 1:
Use the external LLM only in `inference.py`; keep specialists deterministic inside the environment for runtime safety.

---

## 29. Runtime and Infra Constraints

Per brief:
- inference runtime under 20 minutes
- target machine 2 vCPU / 8 GB RAM

Implications:
- keep max steps moderate
- keep observations concise
- avoid nested remote LLM calls in environment runtime
- keep scenario generation light

Recommended max steps:
- easy: 10–15
- medium: 15–20
- hard: 20–25

---

## 30. OpenEnv Compliance Checklist

Must exist and work:
- typed models
- `reset()`
- `step()`
- `state()`
- `openenv.yaml`
- task definitions
- grader definitions
- reproducible baseline

---

## 31. Judge Evaluation Lens

They will effectively judge:
- does it run?
- does it follow spec?
- is it a real task?
- are tasks meaningfully different?
- does reward logic make sense?
- is grading fair and informative?
- can an agent reasonably learn in this world?

They will dislike:
- toy tasks disguised as real
- random rewards
- static tool outputs
- broken Docker / Space setup
- vague README
- incorrect logging format

---

## 32. README Requirements

README must explain:
- what the environment is
- why it is real-world
- action space
- observation space
- task list
- reward design
- grader logic
- setup
- env vars
- local run steps
- deployment notes
- validation steps

---

## 33. Safety and Robustness Policies

- accept one valid tool call per turn
- return structured errors, not crashes
- same seed + same actions => same trajectory
- do not leak hidden grader truth

---

## 34. MVP Scope vs Stretch Scope

Must ship:
- one fully working environment
- 3 tasks
- 12 tools
- single coordinator baseline
- correct logging
- Docker build
- HF Space live
- `openenv validate` passes

Stretch:
- richer analytics
- better route simulation
- improved observation compression
- enhanced grader breakdown

---

## 35. Final Engineering Decisions

- Use 3 specialists + 1 coordinator
- Use 12 tools only
- Keep specialists internal by default
- Use dynamic seeded synthetic scenarios, not mock responses
- Use dense rewards + final graders
- Use rolling memory compression
- Optimize for validator safety first

---

## 36. Execution Order

1. finalize tasks
2. define models
3. implement state
4. build scenario generator
5. build tools
6. build memory + observation logic
7. implement rewards
8. implement graders
9. implement `reset()`, `step()`, `state()`
10. implement `inference.py`
11. write README
12. build Docker
13. deploy to HF Space
14. run validation flow
15. submit Space URL

---

## 37. Final Success Definition

This project is successful if it passes automated validation, runs end-to-end with the baseline LLM, exposes 3 meaningful tasks, uses real state-derived rewards and graders, and feels like a real coordination workflow that an RL policy could learn from.

---

## 38. Source Notes

This master plan incorporates the requirements from the attached Round 1 brief, including OpenEnv compliance, task/grader requirements, inference logging format, validation flow, runtime constraints, mandatory env vars, and Hugging Face Space deployment expectations.
