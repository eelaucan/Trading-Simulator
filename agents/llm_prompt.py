"""Prompt construction for LLM trading agents."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from simulator.config import SimulatorConfig
from simulator.observation import Observation


def build_trading_prompt(observation: Observation, config: SimulatorConfig) -> str:
    """Serialize the visible observation surface into a model-facing prompt."""
    state = observation.portfolio_state
    shares = state.shares_dict()
    market_values = state.market_value_dict()
    nav = max(float(state.total_nav), 1e-12)

    holdings = []
    for ticker in sorted(shares):
        market_value = float(market_values.get(ticker, 0.0))
        holdings.append(
            {
                "ticker": ticker,
                "shares": float(shares[ticker]),
                "market_value": market_value,
                "weight": market_value / nav,
            }
        )

    market_rows = _market_rows(observation)
    price_history = _price_history_rows(observation)
    pending = [
        {
            "ticker": item.ticker,
            "triggered_by_low": float(item.triggered_by_low),
            "stop_level": float(item.stop_level),
            "execution_week": int(item.execution_week) + 1,
        }
        for item in observation.pending_liquidations
    ]

    payload: dict[str, Any] = {
        "instructions": (
            "You are an autonomous long-only weekly trading agent. "
            "Use only the observation below. Do not assume future prices. "
            "Respond with JSON only, using exactly two top-level keys: "
            "'rationale' (string) and 'actions' (array). "
            "Do not repeat the observation back."
        ),
        "decision_week": int(observation.week_index) + 1,
        "as_of": observation.date.isoformat(),
        "portfolio": {
            "cash": float(state.cash),
            "total_nav": float(state.total_nav),
            "weekly_turnover_so_far": float(state.weekly_turnover),
            "concentration_hhi": float(state.concentration_hhi),
            "holdings": holdings,
        },
        "market_this_week": market_rows,
        "visible_price_history": price_history,
        "pending_liquidations": pending,
        "constraints": {
            "long_only": True,
            "max_actions_per_step": int(config.max_actions_per_step),
            "single_stock_cap": float(config.single_stock_cap),
            "hhi_cap": float(config.hhi_cap),
            "weekly_turnover_cap": float(config.turnover_cap),
            "available_tickers": list(observation.available_tickers),
        },
        "action_schema": {
            "actions": [
                {"action_type": "hold"},
                {
                    "action_type": "buy",
                    "ticker": "AAPL",
                    "quantity": 0.05,
                    "quantity_type": "nav_fraction",
                },
                {
                    "action_type": "sell",
                    "ticker": "AAPL",
                    "quantity_type": "close_all",
                },
                {
                    "action_type": "reduce",
                    "ticker": "AAPL",
                    "fraction": 0.5,
                },
                {
                    "action_type": "set_stop",
                    "ticker": "AAPL",
                    "stop_price": 150.0,
                },
                {"action_type": "remove_stop", "ticker": "AAPL"},
            ],
            "quantity_type_values": [
                "shares",
                "notional_dollars",
                "nav_fraction",
                "close_all",
            ],
            "rules": [
                "Submit at most one hold action and never mix hold with other actions.",
                "Prefer nav_fraction for buys when sizing new positions.",
                "Respect turnover and concentration limits; leave cash buffer if uncertain.",
                "If no trade is warranted, return a single hold action.",
            ],
        },
        "response_format": {
            "rationale": "string",
            "actions": "array of action objects matching action_schema",
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _market_rows(observation: Observation) -> list[dict[str, Any]]:
    frame = observation.current_week_ohlcv.copy()
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in frame.sort_values("ticker").iterrows():
        rows.append(
            {
                "ticker": str(row["ticker"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return rows


def _price_history_rows(observation: Observation, max_weeks: int = 12) -> list[dict[str, Any]]:
    history = observation.price_history.copy()
    if history.empty:
        return []
    if "_week_idx" in history.columns:
        history = history.loc[history["_week_idx"] <= observation.week_index]
    if "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"]).dt.strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    for ticker in sorted(observation.available_tickers):
        ticker_frame = history[history["ticker"] == ticker].sort_values("_week_idx").tail(max_weeks)
        for _, row in ticker_frame.iterrows():
            rows.append(
                {
                    "ticker": str(row["ticker"]),
                    "date": str(row.get("date", "")),
                    "close": float(row["close"]),
                }
            )
    return rows
