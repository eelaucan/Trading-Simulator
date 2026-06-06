"""Tests for the Gemini trading agent."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from agents.gemini_agent import GeminiTradingAgent
from simulator.actions import ActionType
from simulator.config import SimulatorConfig
from simulator.observation import Observation
from simulator.state import PortfolioState


class _StubGeminiClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def generate_json(self, *, prompt: str, model: str) -> dict:
        self.calls += 1
        assert "decision_week" in prompt
        return self.payload


def _observation() -> Observation:
    week_index = 12
    current = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "open": [100.0, 50.0],
            "high": [105.0, 52.0],
            "low": [98.0, 49.0],
            "close": [103.0, 51.0],
            "volume": [1_000_000.0, 900_000.0],
            "_week_idx": [week_index, week_index],
        }
    )
    history = pd.DataFrame(
        {
            "date": [datetime(2024, 1, 1, tzinfo=timezone.utc)] * 2,
            "ticker": ["AAA", "BBB"],
            "close": [100.0, 50.0],
            "_week_idx": [week_index, week_index],
        }
    )
    state = PortfolioState(
        week_index=week_index,
        date=datetime(2024, 3, 1, tzinfo=timezone.utc),
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
        date=datetime(2024, 3, 1, tzinfo=timezone.utc),
        current_week_ohlcv=current,
        price_history=history,
        portfolio_state=state,
        available_tickers=["AAA", "BBB"],
        pending_liquidations=[],
    )


def test_gemini_agent_parses_model_actions() -> None:
    client = _StubGeminiClient(
        {
            "rationale": "buy leader",
            "actions": [
                {
                    "action_type": "buy",
                    "ticker": "AAA",
                    "quantity": 0.1,
                    "quantity_type": "nav_fraction",
                }
            ],
        }
    )
    agent = GeminiTradingAgent(
        simulator_config=SimulatorConfig(max_actions_per_step=5),
        client=client,
    )
    actions = agent.decide(_observation())
    assert client.calls == 1
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.BUY
    assert agent.decision_records[0]["used_fallback"] is False


def test_gemini_agent_falls_back_to_hold_on_bad_payload() -> None:
    client = _StubGeminiClient({"actions": "invalid"})
    agent = GeminiTradingAgent(
        simulator_config=SimulatorConfig(max_actions_per_step=5),
        client=client,
    )
    actions = agent.decide(_observation())
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.HOLD
    assert agent.decision_records[0]["used_fallback"] is True


def test_start_gemini_session_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.service import start_session

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        start_session(
            {
                "run_mode": "ai_gemini",
                "dataset_path": "data/sample/weekly_ohlcv_synthetic.csv",
            }
        )
