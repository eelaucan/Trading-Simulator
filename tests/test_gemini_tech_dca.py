"""Tests for the tech DCA Gemini policy."""

from __future__ import annotations

from agents.gemini_tech_dca import TECH_DCA_TICKERS, apply_tech_dca_policy, tech_dca_universe
from agents.runner import run_gemini_agent
from simulator.actions import ActionType
from simulator.config import SimulatorConfig
from simulator.env import TradingEnvironment
from simulator.market import MarketReplay
from agents.gemini_momentum import create_gemini_simulator_config


def test_tech_dca_only_buys_mega_cap_tech() -> None:
    market = MarketReplay("data/sample/weekly_ohlcv_synthetic.csv")
    config = create_gemini_simulator_config(market.available_tickers)
    env = TradingEnvironment(market=market, config=config)
    observation, _ = env.reset()

    actions = apply_tech_dca_policy(observation, config)
    assert actions
    for action in actions:
        if action.action_type == ActionType.BUY:
            assert action.ticker in TECH_DCA_TICKERS


def test_tech_dca_deploys_capital_over_multiple_weeks() -> None:
    result = run_gemini_agent(
        "data/sample/weekly_ohlcv_synthetic.csv",
        output_dir=None,
    )
    nav = float(result.final_state.total_nav)
    tech_weight = sum(
        float(result.final_state.market_value_dict().get(ticker, 0.0)) / nav
        for ticker in TECH_DCA_TICKERS
    )
    assert tech_weight > 0.70
    assert result.final_state.cash / nav < 0.10
    assert any(record.get("decision_source") == "tech_dca" for record in result.agent.decision_records)


def test_tech_dca_universe_filters_unknown_tickers() -> None:
    universe = tech_dca_universe(["AAPL", "JNJ", "NVDA"])
    assert universe == ["NVDA", "AAPL"]
