"""Tests for session-randomized Gemini trading."""

from __future__ import annotations

from agents.gemini_momentum import create_gemini_simulator_config
from agents.gemini_stochastic import (
    GeminiSessionProfile,
    apply_stochastic_momentum_policy,
    create_gemini_session_profile,
)
from agents.benchmark_agent import AutonomousBenchmarkAgent
from agents.runner import run_agent_episode
from simulator.env import TradingEnvironment
from simulator.market import MarketReplay


def _run_profile(profile: GeminiSessionProfile) -> float:
    market = MarketReplay("data/sample/weekly_ohlcv_synthetic.csv")
    config = create_gemini_simulator_config(market.available_tickers)
    config.initial_decision_week = profile.initial_decision_week
    config.seed = profile.session_seed
    env = TradingEnvironment(market=market, config=config)

    class _StochasticAgent:
        def reset_log(self) -> None:
            return None

        def decide(self, observation):
            return apply_stochastic_momentum_policy(observation, config, profile)

    return run_agent_episode(env, _StochasticAgent()).metrics.total_return


def test_stochastic_profiles_vary_returns() -> None:
    profiles = [create_gemini_session_profile(market_weeks=104) for _ in range(8)]
    returns = [_run_profile(profile) for profile in profiles]
    assert all(return_value >= 0.40 for return_value in returns)
    assert len({round(value, 3) for value in returns}) >= 4


def test_session_profile_fields_are_stable() -> None:
    profile = GeminiSessionProfile(
        session_seed=12345,
        temperature=0.82,
        initial_decision_week=51,
    )
    assert profile.should_call_llm(51) == ((51 * 17 + 12345) % 5 == 0)
