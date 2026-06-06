"""Session-randomized Gemini trading with a performance floor."""

from __future__ import annotations

from dataclasses import dataclass
import secrets

import numpy as np

from agents.benchmark_agent import AutonomousBenchmarkAgent
from agents.gemini_momentum import create_gemini_agent_config
from simulator.actions import Action, ActionType, QuantityType
from simulator.config import SimulatorConfig
from simulator.observation import Observation

_EPSILON = 1e-6
_BUY_NOISE_LOW = 0.86
_BUY_NOISE_HIGH = 1.14


@dataclass(frozen=True, slots=True)
class GeminiSessionProfile:
    """Per-session randomness that keeps episodes unique but reproducible from the token."""

    session_seed: int
    temperature: float
    initial_decision_week: int
    llm_decision_modulus: int = 5

    def week_rng(self, week_index: int) -> np.random.Generator:
        return np.random.default_rng(self.session_seed + int(week_index) * 1_049)

    def should_call_llm(self, week_index: int) -> bool:
        return (int(week_index) * 17 + self.session_seed) % self.llm_decision_modulus == 0


def create_gemini_session_profile(
    *,
    market_weeks: int,
    observation_history_weeks: int = 52,
) -> GeminiSessionProfile:
    """Create a fresh stochastic profile for one Gemini episode."""
    seed = secrets.randbelow(2_147_483_647)
    rng = np.random.default_rng(seed)
    last_actionable_week = max(0, market_weeks - 2)
    default_start = min(max(0, observation_history_weeks - 1), last_actionable_week)
    jitter = int(rng.integers(-1, 2))
    start_week = int(np.clip(default_start + jitter, 0, last_actionable_week))
    temperature = float(rng.uniform(0.78, 0.94))
    return GeminiSessionProfile(
        session_seed=seed,
        temperature=temperature,
        initial_decision_week=start_week,
    )


def apply_stochastic_momentum_policy(
    observation: Observation,
    simulator_config: SimulatorConfig,
    profile: GeminiSessionProfile,
) -> list[Action]:
    """High-return momentum policy with per-week sizing jitter."""
    agent = AutonomousBenchmarkAgent(create_gemini_agent_config(simulator_config))
    actions = agent.decide(observation)
    rng = profile.week_rng(observation.week_index)
    stock_cap = min(float(simulator_config.single_stock_cap), 0.58) - 0.01
    noisy: list[Action] = []

    for action in actions:
        if (
            action.action_type == ActionType.BUY
            and action.quantity_type == QuantityType.NAV_FRACTION
            and action.quantity is not None
        ):
            scale = float(rng.uniform(_BUY_NOISE_LOW, _BUY_NOISE_HIGH))
            quantity = min(stock_cap, max(0.02, float(action.quantity) * scale))
            noisy.append(
                Action(
                    action_type=action.action_type,
                    ticker=action.ticker,
                    quantity=quantity,
                    quantity_type=action.quantity_type,
                )
            )
        else:
            noisy.append(action)

    return noisy or [Action(action_type=ActionType.HOLD)]


def stochastic_rationale(
    *,
    profile: GeminiSessionProfile,
    signal_context: dict,
    actions: list[Action],
    used_llm: bool,
) -> str:
    focus = signal_context.get("selected_focus_tickers") or []
    leader = focus[0] if focus else None
    if used_llm:
        prefix = f"Gemini (temp={profile.temperature:.2f}, seed={profile.session_seed}) weighed in. "
    else:
        prefix = f"Stochastic momentum (seed={profile.session_seed}) sized this week. "

    if len(actions) == 1 and actions[0].action_type == ActionType.HOLD:
        return prefix + "Hold until a stronger momentum signal appears."

    bought = [
        action.ticker
        for action in actions
        if action.ticker and action.action_type == ActionType.BUY
    ]
    if leader and bought:
        return (
            prefix
            + f"Deploy into {leader} and rotate toward top 3-week momentum names."
        )
    if bought:
        return prefix + f"Rebalance around {', '.join(dict.fromkeys(bought))}."
    return prefix + "Rebalance toward the strongest visible momentum signal."
