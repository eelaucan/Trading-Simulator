"""JSON serializers for API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import pandas as pd

from simulator.actions import Action, ActionType, QuantityType
from simulator.config import SimulatorConfig
from simulator.env import TradingEnvironment
from simulator.metrics import SimulationMetrics
from simulator.observation import Observation, PendingLiquidation
from simulator.state import PortfolioState
from ui.planner_payload import build_trade_planner_props
from ui.session import SessionMetadata, condition_display_label


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _currency(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def portfolio_state_to_dict(state: PortfolioState) -> dict[str, Any]:
    shares = state.shares_dict()
    market_values = state.market_value_dict()
    cost_basis = state.cost_basis_dict()
    stop_levels = state.stop_levels_dict()
    holdings = []
    for ticker in sorted(shares):
        market_value = float(market_values.get(ticker, 0.0))
        weight = market_value / state.total_nav if state.total_nav > 1e-12 else 0.0
        holdings.append(
            {
                "ticker": ticker,
                "shares": float(shares[ticker]),
                "average_cost": float(cost_basis.get(ticker, 0.0)),
                "market_value": market_value,
                "weight": weight,
                "active_stop": float(stop_levels[ticker]) if ticker in stop_levels else None,
            }
        )
    return {
        "week_index": state.week_index,
        "date": _iso(state.date),
        "cash": float(state.cash),
        "total_nav": float(state.total_nav),
        "invested": max(0.0, float(state.total_nav - state.cash)),
        "positions": sum(1 for quantity in shares.values() if quantity > 1e-12),
        "weekly_turnover": float(state.weekly_turnover),
        "concentration_hhi": float(state.concentration_hhi),
        "portfolio_volatility": state.portfolio_volatility,
        "nav_history": [float(value) for value in state.nav_history],
        "holdings": holdings,
        "allocation": _allocation_rows(state),
    }


def _allocation_rows(state: PortfolioState) -> list[dict[str, Any]]:
    rows = []
    for ticker, value in sorted(state.market_value_dict().items()):
        if value <= 1e-12:
            continue
        rows.append(
            {
                "label": ticker,
                "value": float(value),
                "weight": float(value / state.total_nav) if state.total_nav > 1e-12 else 0.0,
            }
        )
    if state.cash > 1e-12:
        rows.append(
            {
                "label": "Cash",
                "value": float(state.cash),
                "weight": float(state.cash / state.total_nav) if state.total_nav > 1e-12 else 0.0,
            }
        )
    return rows


def pending_liquidations_to_dict(items: Sequence[PendingLiquidation]) -> list[dict[str, Any]]:
    return [
        {
            "ticker": item.ticker,
            "triggered_by_low": float(item.triggered_by_low),
            "stop_level": float(item.stop_level),
            "execution_week": int(item.execution_week) + 1,
        }
        for item in sorted(items, key=lambda value: (value.execution_week, value.ticker))
    ]


def observation_to_dict(observation: Observation) -> dict[str, Any]:
    close_history = observation.price_history.loc[:, ["date", "ticker", "close"]].copy()
    close_history["date"] = pd.to_datetime(close_history["date"]).dt.strftime("%Y-%m-%d")
    current_week = observation.current_week_ohlcv.copy()
    previous_closes = (
        close_history.sort_values("date")
        .groupby("ticker")["close"]
        .apply(lambda series: series.iloc[-2] if len(series) >= 2 else None)
        .to_dict()
    )
    market_rows = []
    for _, row in current_week.iterrows():
        ticker = str(row["ticker"])
        previous = previous_closes.get(ticker)
        change = None
        if previous is not None and float(previous) > 0.0:
            change = float(row["close"]) / float(previous) - 1.0
        market_rows.append(
            {
                "ticker": ticker,
                "close": float(row["close"]),
                "open": float(row["open"]),
                "low": float(row["low"]),
                "high": float(row["high"]),
                "volume": int(row["volume"]),
                "change_vs_previous_close": change,
            }
        )
    return {
        "week_index": int(observation.week_index),
        "date": observation.date.strftime("%Y-%m-%d"),
        "available_tickers": list(observation.available_tickers),
        "pending_liquidations": pending_liquidations_to_dict(observation.pending_liquidations),
        "market_rows": market_rows,
        "price_history": close_history.to_dict(orient="records"),
    }


def metadata_to_dict(metadata: SessionMetadata) -> dict[str, Any]:
    return {
        "participant_id": metadata.participant_id,
        "condition": metadata.condition,
        "condition_label": condition_display_label(metadata.condition),
        "episode_name": metadata.episode_name,
        "dataset_path": metadata.dataset_path,
        "started_at": _iso(metadata.started_at),
        "finished_at": _iso(metadata.finished_at),
        "decision_start_week": metadata.decision_start_week + 1,
        "visible_history_weeks_at_start": metadata.visible_history_weeks_at_start,
        "notes": metadata.notes,
    }


def metrics_to_dict(metrics: SimulationMetrics) -> dict[str, Any]:
    return {
        "total_return": float(metrics.total_return),
        "max_drawdown": float(metrics.max_drawdown),
        "realized_vol": metrics.realized_vol,
        "sharpe_ratio": metrics.sharpe_ratio,
        "avg_hhi": float(metrics.avg_hhi),
        "avg_weekly_turnover": float(metrics.avg_weekly_turnover),
        "blow_up_flag": bool(metrics.blow_up_flag),
        "n_invalid_attempts": int(metrics.n_invalid_attempts),
        "n_clipped_trades": int(metrics.n_clipped_trades),
        "n_stop_triggers": int(metrics.n_stop_triggers),
        "n_gap_adjustments": int(metrics.n_gap_adjustments),
        "vol_rule_activation_week": (
            None
            if metrics.vol_rule_activation_week is None
            else metrics.vol_rule_activation_week + 1
        ),
    }


def actions_from_payload(payload: object, max_actions_per_step: int) -> list[Action]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("Action payload must be a list of planned actions.")

    batch: list[Action] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each planned action must be an object.")
        action = _action_from_payload(item)
        cleaned = [entry for entry in batch if entry.action_type != ActionType.HOLD]
        if len(cleaned) >= max_actions_per_step:
            raise ValueError("The current batch already contains the maximum number of actions.")
        if action.action_type == ActionType.HOLD and cleaned:
            raise ValueError("A do-nothing week must be submitted on its own.")
        batch = list(cleaned)
        batch.append(action)
    return batch


def _action_from_payload(payload: dict[str, Any]) -> Action:
    action_type_raw = payload.get("action_type")
    if not isinstance(action_type_raw, str):
        raise ValueError("Each planned action must include an action_type.")
    action_type = ActionType(action_type_raw)

    ticker_value = payload.get("ticker")
    ticker = ticker_value.strip() if isinstance(ticker_value, str) and ticker_value.strip() else None

    quantity_type_raw = payload.get("quantity_type")
    quantity_type = (
        None
        if quantity_type_raw in (None, "")
        else QuantityType(str(quantity_type_raw))
    )

    quantity_raw = payload.get("quantity")
    quantity = None if quantity_raw in (None, "") else float(quantity_raw)

    fraction_raw = payload.get("fraction")
    fraction = None if fraction_raw in (None, "") else float(fraction_raw)

    stop_price_raw = payload.get("stop_price")
    stop_price = None if stop_price_raw in (None, "") else float(stop_price_raw)

    return Action(
        action_type=action_type,
        ticker=ticker,
        quantity=quantity,
        quantity_type=quantity_type,
        fraction=fraction,
        stop_price=stop_price,
    )


def session_payload(
    *,
    env: TradingEnvironment,
    metadata: SessionMetadata,
    status: str,
    observation: Observation | None,
    state: PortfolioState,
    current_batch: Sequence[Action],
    metrics: SimulationMetrics | None = None,
    last_step_info: dict[str, Any] | None = None,
    error: str | None = None,
    run_mode: str = "human",
    include_planner: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "run_mode": run_mode,
        "metadata": metadata_to_dict(metadata),
        "portfolio": portfolio_state_to_dict(state),
        "done": bool(env.done),
        "error": error,
    }
    if observation is not None:
        payload["observation"] = observation_to_dict(observation)
        if include_planner:
            payload["planner_props"] = build_trade_planner_props(
                config=env.config,
                observation=observation,
                current_batch=current_batch,
            )
    if metrics is not None:
        payload["metrics"] = metrics_to_dict(metrics)
    if last_step_info is not None:
        payload["last_step_info"] = last_step_info
    return payload
