# Village Microgrid Energy Management: Project Understanding

## 1. What the Project is About
This project implements an **OpenEnv-compatible Reinforcement Learning (RL) environment** simulating a **Village Microgrid Energy Management** system. The goal of the project is to provide a standardized, reproducible benchmark where an AI agent acts as a power grid controller for an isolated community. 

The agent operates in a real-time (simulated hourly) scenario and must balance:
- **Available Energy**: Variable solar power (weather dependent), energy stored in a 40 kWh battery, and effectively limitless (but heavily penalized) main grid imports.
- **Power Demand**: "Critical" demand (e.g., hospitals, water pumps) which must never fail, and "Residential" demand (e.g., homes) which can be shed or rationed in an emergency.

At each hour (step), the agent must choose three continuous actions:
1. `grid_import_kwh`: How much energy to buy from the main grid.
2. `battery_action_kwh`: Whether to charge (positive) or discharge (negative) the local battery.
3. `residential_supplied_kwh`: How much of the desired residential load to actually fulfill.

Reward functions heavily penalize unmet critical demand (blackouts) and reward the agent for maintaining power survival. The environment features different scenarios such as a sunny summer day, a dark winter night, and an erratic 72-hour rolling blackout situation.

---

## 2. Is the Environment Static or Dynamic?
The environment acts as a **true dynamic system** (specifically, a Markov Decision Process), but it handles external variables and internal states differently.

### A. The "Static" Components (Exogenous Variables)
The external parameters driving the environment at each time step are predetermined and static:
- **Solar Generation Profiles**: Determined by an array depending on the task (e.g., `_summer_solar_profile()`, `_winter_night_solar_profile()`, or a seeded erratically generated profile).
- **Demand Profiles**: Both `critical_demand` and `residential_demand` follow fixed hourly curves based on daily human behavioral patterns and are statically loaded when an episode is reset.
These variables do not react to the agent's actions (e.g., dropping residential load doesn't change the sun shining).

### B. The "Dynamic" States (Endogenous Variables)
Despite the static profiles, the environment maintains a **dynamic internal state** that is directly evolved by the agent's actions step-by-step. The environment is stateful rather than a static optimization problem.
- **Battery State of Charge (`battery_soc_kwh`)**: This is the primary dynamic state. The battery carries its state from `Step T` to `Step T+1`. If the agent decides to heavily rely on battery discharge early in the day, the `battery_soc` drops, directly constraining what actions are valid or possible later in the episode.
- **Accumulated Outcomes**: The environment tracks cumulative states over the episode based on interacting dynamics, including `_total_reward`, `_critical_failures` (number of blackouts), and `_residential_shed_kwh` (load stripped from homes). 
- **Physics Enforcements applied Dynamically**: If the agent attempts an action that causes a physical energy deficit (e.g., draining more power than exists), the environment dynamically enforces balance by automatically stripping residential supply, and if that isn't enough, forcing an unmet critical blackout state for that turn.

### Summary
The environment operates with **fully dynamic states** bounded by **static external conditions**. Because the battery state of charge bridges one hour to the next, the environment requires sequential decision making and long-term planning, making it a legitimate benchmark for dynamic Reinforcement Learning rather than isolated static predictions.
