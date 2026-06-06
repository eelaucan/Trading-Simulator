"""Tests for Gemini simulator profile."""

from __future__ import annotations

from agents.gemini_momentum import create_gemini_simulator_config


def test_gemini_simulator_config_uses_relaxed_caps() -> None:
    config = create_gemini_simulator_config(["AAA", "BBB"])
    assert config.single_stock_cap == 0.58
    assert config.turnover_cap == 1.0
