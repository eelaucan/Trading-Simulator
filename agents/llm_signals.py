"""Visible-history signal context shared by LLM and benchmark agents."""

from __future__ import annotations

from typing import Any

from agents.benchmark_agent import AutonomousBenchmarkAgent, TickerSignal
from agents.gemini_momentum import create_gemini_agent_config, is_gemini_simulator_config
from agents.runner import create_benchmark_agent_config
from simulator.config import SimulatorConfig
from simulator.observation import Observation


def build_signal_context(
    observation: Observation,
    simulator_config: SimulatorConfig,
) -> dict[str, Any]:
    """Rank tickers using the same momentum/volatility logic as the benchmark agent."""
    if is_gemini_simulator_config(simulator_config):
        agent_config = create_gemini_agent_config(simulator_config)
    else:
        agent_config = create_benchmark_agent_config(simulator_config)
    agent = AutonomousBenchmarkAgent(agent_config)
    signals = agent.compute_signals(observation)
    ranked_positive = _rank_positive_signals(signals, agent_config.min_score)
    selected = ranked_positive[: agent_config.max_positions]

    state = observation.portfolio_state
    nav = max(float(state.total_nav), 1e-12)
    cash_weight = float(state.cash) / nav
    current_weights = {
        ticker: float(value) / nav
        for ticker, value in state.market_value_dict().items()
        if ticker in observation.available_tickers
    }

    target_weights = _target_weights(
        [signal.ticker for signal in selected],
        cash_reserve=agent_config.cash_reserve,
        max_position_weight=agent_config.max_position_weight,
    )

    return {
        "cash_weight": cash_weight,
        "cash_reserve_target": agent_config.cash_reserve,
        "max_positions": agent_config.max_positions,
        "ranked_candidates": [
            {
                "rank": index + 1,
                "ticker": signal.ticker,
                "score": signal.score,
                "momentum_4w": signal.momentum_4w,
                "momentum_12w": signal.momentum_12w,
                "volatility": signal.volatility,
                "latest_close": signal.latest_close,
                "weeks_observed": signal.weeks_observed,
                "current_weight": current_weights.get(signal.ticker, 0.0),
                "target_weight": target_weights.get(signal.ticker, 0.0),
            }
            for index, signal in enumerate(ranked_positive[:8])
        ],
        "selected_focus_tickers": [signal.ticker for signal in selected],
        "suggested_target_weights": target_weights,
        "held_tickers": [
            ticker
            for ticker, weight in sorted(current_weights.items())
            if weight > 1e-6
        ],
    }


def has_actionable_candidates(signal_context: dict[str, Any]) -> bool:
    return bool(signal_context.get("selected_focus_tickers"))


def _rank_positive_signals(
    signals: list[TickerSignal],
    min_score: float,
) -> list[TickerSignal]:
    positive = [
        signal
        for signal in signals
        if signal.eligible and signal.score is not None and signal.score > min_score
    ]
    return sorted(positive, key=lambda signal: (-float(signal.score or 0.0), signal.ticker))


def _target_weights(
    selected_tickers: list[str],
    *,
    cash_reserve: float,
    max_position_weight: float,
) -> dict[str, float]:
    if not selected_tickers:
        return {}
    investable_weight = max(0.0, 1.0 - cash_reserve)
    equal_weight = investable_weight / float(len(selected_tickers))
    per_position = min(max_position_weight, equal_weight)
    return {ticker: float(per_position) for ticker in selected_tickers}
