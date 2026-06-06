"""Concentrated momentum policy tuned for Gemini AI sessions."""

from __future__ import annotations

from typing import Any

from agents.benchmark_agent import AutonomousBenchmarkAgent, BenchmarkAgentConfig
from simulator.actions import Action, ActionType
from simulator.config import SimulatorConfig
from simulator.observation import Observation


_GEMINI_SINGLE_STOCK_CAP = 0.58
_GEMINI_TURNOVER_CAP = 1.0


def is_gemini_simulator_config(config: SimulatorConfig) -> bool:
    """Whether the active simulator rules match the Gemini AI session profile."""
    return (
        float(config.single_stock_cap) >= _GEMINI_SINGLE_STOCK_CAP - 1e-6
        and float(config.turnover_cap) >= _GEMINI_TURNOVER_CAP - 1e-6
    )


def create_gemini_simulator_config(ticker_universe: list[str]) -> SimulatorConfig:
    """Simulator limits that allow fast deployment into top momentum names."""
    return SimulatorConfig(
        initial_cash=100_000.0,
        ticker_universe=ticker_universe,
        single_stock_cap=_GEMINI_SINGLE_STOCK_CAP,
        hhi_cap=_GEMINI_SINGLE_STOCK_CAP,
        turnover_cap=_GEMINI_TURNOVER_CAP,
        cash_buffer=0.0,
        max_actions_per_step=8,
    )


def create_gemini_agent_config(simulator_config: SimulatorConfig) -> BenchmarkAgentConfig:
    """Aggressive single-name momentum rotation aligned with Gemini simulator caps."""
    cap_buffer = 0.01
    stock_cap = min(float(simulator_config.single_stock_cap), _GEMINI_SINGLE_STOCK_CAP) - cap_buffer
    return BenchmarkAgentConfig(
        cash_reserve=0.01,
        max_position_weight=max(0.01, stock_cap),
        max_turnover=float(simulator_config.turnover_cap),
        max_positions=1,
        max_actions_per_step=int(simulator_config.max_actions_per_step),
        short_momentum_window=3,
        medium_momentum_window=8,
        short_momentum_weight=1.0,
        medium_momentum_weight=0.0,
        volatility_penalty=0.0,
        rebalance_threshold=0.0,
    )


def apply_gemini_momentum_policy(
    observation: Observation,
    simulator_config: SimulatorConfig,
) -> list[Action]:
    """Return weekly actions from the concentrated momentum policy."""
    agent = AutonomousBenchmarkAgent(create_gemini_agent_config(simulator_config))
    return agent.decide(observation)


def momentum_rationale(signal_context: dict[str, Any], actions: list[Action]) -> str:
    """Human-readable summary for decision logs and the finished screen."""
    focus = signal_context.get("selected_focus_tickers") or []
    leader = focus[0] if focus else None
    if not actions:
        return "No trades generated this week."
    if len(actions) == 1 and actions[0].action_type == ActionType.HOLD:
        return "Hold cash until a positive 3-week momentum leader appears."
    traded = [
        action.ticker
        for action in actions
        if action.ticker and action.action_type in {ActionType.BUY, ActionType.SELL}
    ]
    if leader and traded:
        return (
            f"Rotate toward {leader}, the top 3-week momentum name, "
            f"using up to {_GEMINI_SINGLE_STOCK_CAP:.0%} single-stock capacity."
        )
    if traded:
        return f"Rebalance around {', '.join(dict.fromkeys(traded))}."
    return "Rebalance toward the strongest visible momentum signal."
