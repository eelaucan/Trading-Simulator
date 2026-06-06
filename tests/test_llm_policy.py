"""Tests for LLM signal rescue policy."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from agents.gemini_agent import GeminiTradingAgent
from agents.llm_policy import should_apply_signal_rescue
from agents.runner import run_agent_episode
from simulator.actions import Action, ActionType
from simulator.config import SimulatorConfig
from simulator.env import TradingEnvironment
from simulator.market import MarketReplay
from simulator.observation import Observation
from simulator.state import PortfolioState


class _HoldOnlyClient:
    def generate_json(self, *, prompt: str, model: str) -> dict:
        return {"rationale": "stay cautious", "actions": [{"action_type": "hold"}]}


def _observation_with_history(week_index: int = 51) -> Observation:
    tickers = ["AAA", "BBB", "CCC"]
    rows = []
    for ticker in tickers:
        for offset, close in enumerate([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0, 122.0, 124.0]):
            rows.append(
                {
                    "ticker": ticker,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close + (2.0 if ticker == "AAA" else 0.5),
                    "volume": 1_000_000.0,
                    "_week_idx": week_index - 12 + offset,
                    "date": datetime(2024, 1, 1, tzinfo=timezone.utc),
                }
            )
    history = pd.DataFrame(rows)
    current = history[history["_week_idx"] == week_index].copy()
    state = PortfolioState(
        week_index=week_index,
        date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        cash=100_000.0,
        shares=tuple(),
        market_value=tuple(),
        total_nav=100_000.0,
        realized_pnl=0.0,
        unrealized_pnl=tuple(),
        cost_basis=tuple(),
        stop_levels=tuple(),
        weekly_turnover=0.0,
        concentration_hhi=0.0,
        portfolio_volatility=None,
        nav_history=(100_000.0,),
    )
    return Observation(
        week_index=week_index,
        date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        current_week_ohlcv=current,
        price_history=history,
        portfolio_state=state,
        available_tickers=tickers,
        pending_liquidations=[],
    )


def test_should_apply_signal_rescue_for_hold_with_high_cash() -> None:
    from agents.llm_signals import build_signal_context

    observation = _observation_with_history()
    config = SimulatorConfig(max_actions_per_step=5, ticker_universe=["AAA", "BBB", "CCC"])
    context = build_signal_context(observation, config)
    assert should_apply_signal_rescue(observation, [Action(action_type=ActionType.HOLD)], context)


def test_gemini_batch_advances_multiple_weeks(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from api.service import advance_gemini_session_batch, start_session

    session = start_session(
        {
            "run_mode": "ai_gemini",
            "participant_id": "batch_test",
            "episode_name": "pilot",
            "dataset_path": "data/sample/weekly_ohlcv_synthetic.csv",
        }
    )
    response = advance_gemini_session_batch(session["session"], max_steps=3, time_budget_seconds=20)
    assert response["batch_steps"] == 3
    assert response["status"] == "running"
    assert "planner_props" not in response


def test_hold_only_client_runs_full_episode_with_signal_rescue() -> None:
    from agents.gemini_momentum import create_gemini_simulator_config

    market = MarketReplay("data/sample/weekly_ohlcv_synthetic.csv")
    config = create_gemini_simulator_config(market.available_tickers)
    env = TradingEnvironment(market=market, config=config)
    agent = GeminiTradingAgent(simulator_config=config, client=_HoldOnlyClient())
    result = run_agent_episode(env, agent)
    assert result.metrics.total_return >= 0.40
    assert any(
        record.get("decision_source") in {"signal_rescue", "momentum_agent"}
        for record in agent.decision_records
    )
