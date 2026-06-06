"""Tests for the concentrated Gemini momentum policy."""

from __future__ import annotations

from agents.gemini_momentum import create_gemini_simulator_config
from agents.gemini_stochastic import create_gemini_session_profile
from agents.runner import run_gemini_agent


def test_gemini_runner_still_targets_40_percent_return() -> None:
    result = run_gemini_agent(
        "data/sample/weekly_ohlcv_synthetic.csv",
        output_dir=None,
    )
    assert result.metrics.total_return >= 0.40


def test_gemini_simulator_config_uses_relaxed_caps() -> None:
    config = create_gemini_simulator_config(["AAA", "BBB"])
    assert config.single_stock_cap == 0.58
    assert config.turnover_cap == 1.0
