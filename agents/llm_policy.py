"""Post-processing rules for LLM trading decisions."""

from __future__ import annotations

from agents.benchmark_agent import AutonomousBenchmarkAgent
from agents.llm_signals import build_signal_context, has_actionable_candidates
from agents.gemini_momentum import create_gemini_agent_config, is_gemini_simulator_config
from agents.runner import create_benchmark_agent_config
from simulator.actions import Action, ActionType, QuantityType
from simulator.config import SimulatorConfig
from simulator.observation import Observation


def should_apply_signal_rescue(
    observation: Observation,
    actions: list[Action],
    signal_context: dict,
) -> bool:
    """Use the benchmark policy when the model under-invests despite strong signals."""
    if not has_actionable_candidates(signal_context):
        return False

    state = observation.portfolio_state
    nav = max(float(state.total_nav), 1e-12)
    cash_weight = float(state.cash) / nav
    if cash_weight < 0.35:
        return False

    if _is_hold_only(actions):
        return True

    buy_weight = _planned_buy_weight(actions)
    return buy_weight < 0.08


def apply_signal_rescue(
    observation: Observation,
    simulator_config: SimulatorConfig,
) -> list[Action]:
    """Return benchmark-agent actions derived from the same visible signals."""
    if is_gemini_simulator_config(simulator_config):
        agent_config = create_gemini_agent_config(simulator_config)
    else:
        agent_config = create_benchmark_agent_config(simulator_config)
    agent = AutonomousBenchmarkAgent(agent_config)
    return agent.decide(observation)


def _is_hold_only(actions: list[Action]) -> bool:
    return len(actions) == 1 and actions[0].action_type == ActionType.HOLD


def _planned_buy_weight(actions: list[Action]) -> float:
    total = 0.0
    for action in actions:
        if action.action_type != ActionType.BUY:
            continue
        if action.quantity_type == QuantityType.NAV_FRACTION and action.quantity is not None:
            total += max(0.0, float(action.quantity))
    return total
