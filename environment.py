"""
Village Microgrid Energy Management — OpenEnv Environment
===========================================================
The agent acts as the microgrid controller for a small village.
Goal: balance power generation, battery storage, and energy consumption
over a 24-hour (or 72-hour for Task 3) simulation where 1 step = 1 hour.

Physics model: simple additive kWh arithmetic (no complex electrical formulas).

Energy balance per step
-----------------------
  available_kwh = solar_gen + grid_import + battery_discharge
  used_kwh      = critical_demand + residential_supplied + battery_charge
  surplus_kwh   = available_kwh - used_kwh  (spilled if positive)

Any deficit is resolved by first cutting residential supply, then flagging
unmet critical demand (worst-case blackout penalty).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Models (OpenEnv Spec)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent observes at the start of each hour-step."""

    step_hour: int = Field(
        ..., ge=0, le=71,
        description="Current simulation hour (0-based; for 24-h tasks, range 0-23)."
    )
    solar_gen_kwh: float = Field(
        ..., ge=0.0,
        description="Solar generation available this hour (kWh)."
    )
    battery_soc_kwh: float = Field(
        ..., ge=0.0,
        description="Battery state-of-charge at the start of the step (kWh)."
    )
    critical_demand_kwh: float = Field(
        ..., ge=0.0,
        description="Non-negotiable demand (hospital, water pump) that must be met (kWh)."
    )
    residential_demand_kwh: float = Field(
        ..., ge=0.0,
        description="Standard residential demand that can be shed in an emergency (kWh)."
    )


class Action(BaseModel):
    """The agent's decisions for a single hour step."""

    grid_import_kwh: float = Field(
        ..., ge=0.0,
        description="Power to purchase from the main grid this hour (kWh). Each kWh costs a reward penalty."
    )
    battery_action_kwh: float = Field(
        ...,
        description=(
            "Battery action (kWh). "
            "Positive  → charge battery (consumes available energy). "
            "Negative  → discharge battery (releases stored energy). "
            "Zero      → battery idle."
        ),
    )
    residential_supplied_kwh: float = Field(
        ..., ge=0.0,
        description=(
            "How much of the residential demand the agent chooses to fulfil (kWh). "
            "Must be ≤ residential_demand_kwh observed in the previous step."
        )
    )


class Reward(BaseModel):
    """Structured reward breakdown (returned inside info dict)."""

    total: float = Field(..., description="Net reward for this step.")
    base: float = Field(default=1.0, description="Base survival reward (+1.0 per step).")
    critical_penalty: float = Field(
        default=0.0,
        description="Penalty for unmet critical demand (unmet_kWh × -5.0)."
    )
    residential_penalty: float = Field(
        default=0.0,
        description="Penalty for unmet residential demand (unmet_kWh × -0.5)."
    )
    grid_penalty: float = Field(
        default=0.0,
        description="Penalty for grid imports (grid_kWh × -0.1)."
    )
    invalid_action_penalty: float = Field(
        default=0.0,
        description="Penalty for physically invalid actions (-1.0 each)."
    )


class EnvironmentState(BaseModel):
    """Full internal environment snapshot returned by state()."""

    current_step: int
    max_steps: int
    battery_soc_kwh: float
    battery_capacity_kwh: float
    task_name: str
    total_reward: float
    critical_failures: int
    residential_shed_kwh: float


# ---------------------------------------------------------------------------
# Solar Generation Profiles
# ---------------------------------------------------------------------------

def _summer_solar_profile() -> List[float]:
    """
    24-hour solar profile for a perfect summer day.
    Peak ~8.5 kWh at mid-day; zero overnight.
    """
    return [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.5,   # 00:00 – 05:00
        1.5, 3.0, 5.0, 7.0, 8.0, 8.5,   # 06:00 – 11:00
        8.0, 7.5, 6.5, 5.0, 3.5, 1.5,   # 12:00 – 17:00
        0.5, 0.0, 0.0, 0.0, 0.0, 0.0,   # 18:00 – 23:00
    ]


def _winter_night_solar_profile() -> List[float]:
    """
    24-hour solar profile for a winter near-blackout scenario.
    Very limited solar — effectively near-zero most of the day.
    Agent must rely heavily on battery reserves and grid imports.
    """
    return [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.2, 0.5, 0.8, 1.0, 1.0, 0.8,
        0.5, 0.3, 0.2, 0.1, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]


def _erratic_solar_profile(seed: int = 42) -> List[float]:
    """
    72-hour erratic solar profile for the rolling-blackout task.
    Takes a base summer profile and applies random weather multipliers
    to simulate sudden cloud cover, storms, and brief sunny spells.

    Seeded for reproducibility across runs.
    """
    rng = random.Random(seed)
    base_24 = _summer_solar_profile()
    profile: List[float] = []
    for _ in range(3):  # 3 days
        for hour_val in base_24:
            weather = rng.choices(
                population=[0.0, 0.3, 0.7, 1.0, 1.2],
                weights=[0.10, 0.15, 0.25, 0.35, 0.15],
            )[0]
            profile.append(round(hour_val * weather, 2))
    return profile


# ---------------------------------------------------------------------------
# Demand Profiles
# ---------------------------------------------------------------------------

def _base_critical_demand(hours: int = 24) -> List[float]:
    """
    Critical demand (hospital + water pump) over `hours` steps.
    Roughly constant at ~2.0–2.5 kWh/h.
    Repeats the 24-h base pattern for multi-day tasks.
    """
    critical_24 = [
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0,   # night
        2.5, 2.5, 2.5, 2.5, 2.5, 2.5,   # morning
        2.5, 2.5, 2.5, 2.5, 2.5, 2.5,   # afternoon
        2.5, 2.5, 2.0, 2.0, 2.0, 2.0,   # evening
    ]
    return (critical_24 * ((hours // 24) + 1))[:hours]


def _base_residential_demand(hours: int = 24) -> List[float]:
    """
    Residential demand over `hours` steps.
    Peaks in the morning (commute prep) and evening (dinner/entertainment).
    """
    res_24 = [
        1.0, 0.8, 0.8, 0.8, 1.0, 1.5,   # night → early morning
        3.0, 4.0, 3.5, 2.5, 2.0, 2.0,   # morning peak → midday
        2.5, 2.0, 2.0, 2.5, 3.5, 5.0,   # afternoon → evening ramp
        5.5, 5.0, 4.0, 3.0, 2.0, 1.5,   # evening peak → late night
    ]
    return (res_24 * ((hours // 24) + 1))[:hours]


# ---------------------------------------------------------------------------
# Grader Classes
# ---------------------------------------------------------------------------

class Task1Grader:
    """
    Task 1 – Perfect Summer Day Grader
    ------------------------------------
    Score = 1.0 iff:
      • 100 % of critical AND residential demand is met every step, AND
      • Battery ends the day at ≥ 90 % capacity.

    Score degrades linearly with unmet demand and battery shortfall.
    Any unmet critical demand → automatic 0.0.
    """

    def grade(
        self,
        info_history: List[Dict[str, Any]],
        final_battery_soc: float,
        battery_capacity: float,
    ) -> float:
        if not info_history:
            return 0.0

        total_critical_unmet = sum(s.get("unmet_critical_kwh", 0.0) for s in info_history)
        if total_critical_unmet > 0.01:
            return 0.0  # Automatic fail — hospital went dark

        # Battery end-of-day: ≥ 90 % → full marks on this sub-score
        battery_score = min(final_battery_soc / (0.9 * battery_capacity), 1.0)

        # Residential service score
        total_residential = sum(s.get("residential_demand_kwh", 0.0) for s in info_history)
        total_residential_unmet = sum(s.get("unmet_residential_kwh", 0.0) for s in info_history)
        if total_residential > 0:
            residential_score = max(0.0, 1.0 - total_residential_unmet / total_residential)
        else:
            residential_score = 1.0

        return round(0.5 * battery_score + 0.5 * residential_score, 4)


class Task2Grader:
    """
    Task 2 – Winter Night Grader
    ------------------------------
    Automatic 0.0 if ANY step has unmet critical demand.
    Score = 1 − (total_grid_import / reference_max_import)
    Rewards self-sufficiency — heavy grid reliance lowers the score.
    """

    # Theoretical worst-case: buying the full import headroom every step
    MAX_GRID_REFERENCE_KWH: float = 200.0

    def grade(self, info_history: List[Dict[str, Any]]) -> float:
        if not info_history:
            return 0.0
        for step_info in info_history:
            if step_info.get("unmet_critical_kwh", 0.0) > 0.01:
                return 0.0  # Automatic fail

        total_grid = sum(s.get("grid_import_kwh", 0.0) for s in info_history)
        grid_ratio = min(total_grid / self.MAX_GRID_REFERENCE_KWH, 1.0)
        return round(max(0.0, 1.0 - grid_ratio), 4)


class Task3Grader:
    """
    Task 3 – Rolling Blackout (72 hours) Grader
    ---------------------------------------------
    Composite score:
      • 60 % weight → critical uptime fraction
          (fraction of 72 steps where unmet_critical_kwh ≤ 0.01)
      • 40 % weight → residential service fraction
          (fraction of total residential demand actually served)

    Unlike Tasks 1 & 2, a single critical failure does NOT auto-zero the score
    (because 72 h makes total avoidance very hard) — but it still hurts a lot.
    """

    def grade(self, info_history: List[Dict[str, Any]]) -> float:
        if not info_history:
            return 0.0

        total_steps = len(info_history)
        critical_ok = sum(
            1 for s in info_history if s.get("unmet_critical_kwh", 0.0) <= 0.01
        )
        uptime_score = critical_ok / total_steps

        total_res_demanded = sum(s.get("residential_demand_kwh", 0.0) for s in info_history)
        total_res_unmet = sum(s.get("unmet_residential_kwh", 0.0) for s in info_history)
        if total_res_demanded > 0:
            residential_score = max(0.0, 1.0 - total_res_unmet / total_res_demanded)
        else:
            residential_score = 1.0

        return round(0.6 * uptime_score + 0.4 * residential_score, 4)


# ---------------------------------------------------------------------------
# Main Environment Class
# ---------------------------------------------------------------------------

class VillageMicrogridEnv:
    """
    Village Microgrid Energy Management — OpenEnv-compliant environment.

    The agent controls a small village microgrid over a 24-hour (Tasks 1 & 2)
    or 72-hour (Task 3) horizon.  Each step represents 1 hour.

    At every step the agent chooses:
      1. ``grid_import_kwh``        — how much to buy from the main grid (costly).
      2. ``battery_action_kwh``     — charge (+) or discharge (−) the battery.
      3. ``residential_supplied_kwh``— how much residential demand to fulfil.

    Reward shaping (per step):
      +1.0   base survival reward
      −5.0 × unmet_critical_kwh    (blackout penalty — hospital/water)
      −0.5 × unmet_residential_kwh (load-shedding penalty)
      −0.1 × grid_import_kwh       (self-sufficiency incentive)
      −1.0 × invalid_action_count  (physics violation)

    Hardware constants
    ------------------
    Battery capacity     : 40 kWh
    Max charge rate      : 10 kWh/step
    Max discharge rate   : 10 kWh/step
    Task 3 grid cap      :  3 kWh/step (hard cap, enforced automatically)
    """

    DEFAULT_BATTERY_CAPACITY_KWH: float = 40.0
    DEFAULT_BATTERY_MAX_CHARGE_RATE: float = 10.0
    DEFAULT_BATTERY_MAX_DISCHARGE_RATE: float = 10.0
    TASK3_GRID_CAP_KWH: float = 3.0

    SUPPORTED_TASKS: List[str] = [
        "task1_summer_day",
        "task2_winter_night",
        "task3_rolling_blackout",
    ]

    def __init__(self) -> None:
        """Initialise hardware limits.  Call ``reset(task_name)`` before use."""
        self.battery_capacity_kwh: float = self.DEFAULT_BATTERY_CAPACITY_KWH
        self.battery_max_charge_rate: float = self.DEFAULT_BATTERY_MAX_CHARGE_RATE
        self.battery_max_discharge_rate: float = self.DEFAULT_BATTERY_MAX_DISCHARGE_RATE

        # Runtime state (populated by reset)
        self._task_name: str = ""
        self._max_steps: int = 24
        self._current_step: int = 0
        self._battery_soc: float = 0.0
        self._solar_profile: List[float] = []
        self._critical_demand_profile: List[float] = []
        self._residential_demand_profile: List[float] = []
        self._info_history: List[Dict[str, Any]] = []
        self._total_reward: float = 0.0
        self._critical_failures: int = 0
        self._residential_shed_kwh: float = 0.0
        self._task3_grid_cap: Optional[float] = None

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "task1_summer_day") -> Observation:
        """
        Initialise (or re-initialise) the environment for a specific task.

        Parameters
        ----------
        task_name : str
            One of ``'task1_summer_day'``, ``'task2_winter_night'``,
            ``'task3_rolling_blackout'``.

        Returns
        -------
        Observation
            The initial observation before any action is taken (step 0).

        Raises
        ------
        ValueError
            If ``task_name`` is not in ``SUPPORTED_TASKS``.
        """
        if task_name not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from {self.SUPPORTED_TASKS}."
            )

        self._task_name = task_name
        self._current_step = 0
        self._info_history = []
        self._total_reward = 0.0
        self._critical_failures = 0
        self._residential_shed_kwh = 0.0
        self._task3_grid_cap = None

        if task_name == "task1_summer_day":
            # Easy: lots of solar, low demand, battery starts EMPTY (use solar to fill it)
            self._max_steps = 24
            self._solar_profile = _summer_solar_profile()
            self._critical_demand_profile = _base_critical_demand(24)
            self._residential_demand_profile = _base_residential_demand(24)
            self._battery_soc = 0.0

        elif task_name == "task2_winter_night":
            # Medium: minimal solar, battery starts FULL, careful grid import needed
            self._max_steps = 24
            self._solar_profile = _winter_night_solar_profile()
            self._critical_demand_profile = _base_critical_demand(24)
            self._residential_demand_profile = _base_residential_demand(24)
            self._battery_soc = self.battery_capacity_kwh  # Full battery

        elif task_name == "task3_rolling_blackout":
            # Hard: 72 hours, erratic solar, hard per-step grid import cap
            self._max_steps = 72
            self._solar_profile = _erratic_solar_profile(seed=42)
            self._critical_demand_profile = _base_critical_demand(72)
            self._residential_demand_profile = _base_residential_demand(72)
            self._battery_soc = self.battery_capacity_kwh * 0.5  # Half-charged
            self._task3_grid_cap = self.TASK3_GRID_CAP_KWH

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Advance the simulation by one hour.

        Energy balance logic
        --------------------
        1. Clamp and validate action values against hardware limits.
        2. Compute available energy: solar + grid_import + battery_discharge.
        3. Compute required energy: critical_demand + residential_supplied + battery_charge.
        4. If available < required (deficit):
             a. Reduce residential_supplied to cover the shortfall first.
             b. If still in deficit, flag remaining gap as unmet_critical.
        5. Update battery SoC.
        6. Compute and return reward.

        Parameters
        ----------
        action : Action
            The agent's decision for this hour.

        Returns
        -------
        obs : Observation
            Observation at the START of the next step.
        reward : float
            Scalar step reward (can be negative).
        done : bool
            True when the episode has reached ``max_steps``.
        info : dict
            Diagnostic dictionary with full energy-flow breakdown.
        """
        assert self._task_name, "Call reset(task_name) before step()."
        assert self._current_step < self._max_steps, "Episode is already done. Call reset()."

        solar = self._solar_profile[self._current_step]
        critical_demand = self._critical_demand_profile[self._current_step]
        residential_demand = self._residential_demand_profile[self._current_step]

        # --- Sanitise action values (no crashes on bad LLM output) ---
        grid_import = max(0.0, action.grid_import_kwh)
        battery_action = action.battery_action_kwh          # + charge, − discharge
        residential_supplied = max(
            0.0, min(action.residential_supplied_kwh, residential_demand)
        )

        invalid_action_count = 0

        # Enforce battery charge rate / capacity ceiling
        if battery_action > 0:
            max_possible_charge = min(
                self.battery_max_charge_rate,
                self.battery_capacity_kwh - self._battery_soc,
            )
            if battery_action > max_possible_charge + 0.01:
                invalid_action_count += 1
            battery_action = min(battery_action, max_possible_charge)

        # Enforce battery discharge rate / floor
        elif battery_action < 0:
            max_possible_discharge = min(
                self.battery_max_discharge_rate, self._battery_soc
            )
            if abs(battery_action) > max_possible_discharge + 0.01:
                invalid_action_count += 1
            battery_action = -min(abs(battery_action), max_possible_discharge)

        # Enforce Task 3 grid import hard cap
        if self._task3_grid_cap is not None and grid_import > self._task3_grid_cap:
            invalid_action_count += 1
            grid_import = self._task3_grid_cap

        # --- Energy balance ---
        battery_discharge = max(0.0, -battery_action)   # kWh released from battery
        battery_charge = max(0.0, battery_action)        # kWh going INTO battery

        available_kwh = solar + grid_import + battery_discharge
        required_kwh = critical_demand + residential_supplied + battery_charge

        surplus = available_kwh - required_kwh   # positive → spilled; negative → deficit

        unmet_critical = 0.0
        unmet_residential = 0.0

        if surplus < -0.01:  # Non-trivial deficit
            deficit = abs(surplus)
            # Priority 1: cut residential supply to cover deficit
            residential_cut = min(residential_supplied, deficit)
            residential_supplied = max(0.0, residential_supplied - residential_cut)
            deficit -= residential_cut
            # Priority 2: if still in deficit, critical demand is unmet (blackout)
            if deficit > 0.01:
                unmet_critical = deficit

        unmet_residential = max(0.0, residential_demand - residential_supplied)

        # --- Update battery SoC ---
        self._battery_soc = max(
            0.0,
            min(self.battery_capacity_kwh, self._battery_soc + battery_action),
        )

        # --- Reward calculation ---
        reward_components = Reward(
            total=0.0,
            base=1.0,
            critical_penalty=-5.0 * unmet_critical,
            residential_penalty=-0.5 * unmet_residential,
            grid_penalty=-0.1 * grid_import,
            invalid_action_penalty=-1.0 * invalid_action_count,
        )
        step_reward = (
            reward_components.base
            + reward_components.critical_penalty
            + reward_components.residential_penalty
            + reward_components.grid_penalty
            + reward_components.invalid_action_penalty
        )
        reward_components.total = step_reward

        self._total_reward += step_reward
        if unmet_critical > 0.01:
            self._critical_failures += 1
        self._residential_shed_kwh += unmet_residential

        # --- Info dict ---
        surplus_spilled = max(
            0.0,
            available_kwh - critical_demand - residential_supplied - battery_charge,
        )
        info: Dict[str, Any] = {
            "step": self._current_step,
            "solar_gen_kwh": solar,
            "grid_import_kwh": grid_import,
            "battery_action_kwh": battery_action,
            "battery_soc_after_kwh": self._battery_soc,
            "critical_demand_kwh": critical_demand,
            "residential_demand_kwh": residential_demand,
            "residential_supplied_kwh": residential_supplied,
            "unmet_critical_kwh": round(unmet_critical, 4),
            "unmet_residential_kwh": round(unmet_residential, 4),
            "surplus_spilled_kwh": round(surplus_spilled, 4),
            "step_reward": round(step_reward, 4),
            "reward_breakdown": reward_components.model_dump(),
            "invalid_actions": invalid_action_count,
        }
        self._info_history.append(info)

        self._current_step += 1
        done = self._current_step >= self._max_steps
        next_obs = self._get_observation() if not done else self._get_terminal_observation()
        return next_obs, step_reward, done, info

    def state(self) -> Dict[str, Any]:
        """
        Return the full internal state of the environment as a plain dictionary.

        Useful for debugging, logging, and OpenEnv spec compliance checks.
        """
        return EnvironmentState(
            current_step=self._current_step,
            max_steps=self._max_steps,
            battery_soc_kwh=self._battery_soc,
            battery_capacity_kwh=self.battery_capacity_kwh,
            task_name=self._task_name,
            total_reward=self._total_reward,
            critical_failures=self._critical_failures,
            residential_shed_kwh=round(self._residential_shed_kwh, 4),
        ).model_dump()

    # ------------------------------------------------------------------
    # Grading API
    # ------------------------------------------------------------------

    def grade_episode(self) -> float:
        """
        Score the completed episode using the task-specific grader.

        Must be called AFTER the episode is done (``done == True``).

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if not self._info_history:
            return 0.0

        if self._task_name == "task1_summer_day":
            grader = Task1Grader()
            return grader.grade(
                info_history=self._info_history,
                final_battery_soc=self._battery_soc,
                battery_capacity=self.battery_capacity_kwh,
            )
        if self._task_name == "task2_winter_night":
            grader = Task2Grader()
            return grader.grade(info_history=self._info_history)
        if self._task_name == "task3_rolling_blackout":
            grader = Task3Grader()
            return grader.grade(info_history=self._info_history)
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> Observation:
        """Build observation from current step index (pre-action view)."""
        if self._current_step >= self._max_steps:
            return self._get_terminal_observation()
        return Observation(
            step_hour=self._current_step % 24,
            solar_gen_kwh=self._solar_profile[self._current_step],
            battery_soc_kwh=self._battery_soc,
            critical_demand_kwh=self._critical_demand_profile[self._current_step],
            residential_demand_kwh=self._residential_demand_profile[self._current_step],
        )

    def _get_terminal_observation(self) -> Observation:
        """Return a zeroed-out observation to signal episode end."""
        return Observation(
            step_hour=self._current_step % 24,
            solar_gen_kwh=0.0,
            battery_soc_kwh=self._battery_soc,
            critical_demand_kwh=0.0,
            residential_demand_kwh=0.0,
        )


# ---------------------------------------------------------------------------
# Quick sanity-check (run directly: python environment.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = VillageMicrogridEnv()
    for task in VillageMicrogridEnv.SUPPORTED_TASKS:
        obs = env.reset(task)
        done = False
        while not done:
            # Greedy heuristic: supply all critical + all residential, use solar first
            discharge = min(obs.battery_soc_kwh, env.battery_max_discharge_rate)
            needed = obs.critical_demand_kwh + obs.residential_demand_kwh
            available = obs.solar_gen_kwh + discharge
            grid_need = max(0.0, needed - available)
            act = Action(
                grid_import_kwh=round(grid_need, 2),
                battery_action_kwh=round(-discharge if discharge > 0 else 0.0, 2),
                residential_supplied_kwh=round(obs.residential_demand_kwh, 2),
            )
            obs, reward, done, info = env.step(act)
        score = env.grade_episode()
        print(f"{task:35s} | total_reward={env._total_reward:7.2f} | score={score:.4f}")
