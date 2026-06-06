"""Planner payload builders without Streamlit or chart dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from simulator.actions import Action, ActionType, QuantityType
from simulator.config import SimulatorConfig
from simulator.observation import Observation


_EPSILON = 1e-12
_PREVIEW_ACTION_ORDER: dict[ActionType, int] = {
    ActionType.SELL: 0,
    ActionType.REDUCE: 0,
    ActionType.BUY: 1,
    ActionType.SET_STOP: 2,
    ActionType.REMOVE_STOP: 2,
    ActionType.HOLD: 3,
}


@dataclass(frozen=True, slots=True)
class PlanImpactPreview:
    estimated_spend: float
    estimated_proceeds: float
    estimated_transaction_costs: float
    estimated_remaining_cash: float
    estimated_positions_after: int
    estimated_invested_after: float
    projected_max_weight: float | None
    warnings: tuple[str, ...]
    notes: tuple[str, ...]


def build_trade_planner_props(
    *,
    config: SimulatorConfig,
    observation: Observation,
    current_batch: Sequence[Action],
) -> dict[str, Any]:
    state = observation.portfolio_state
    shares = state.shares_dict()
    market_values = state.market_value_dict()
    cost_basis = state.cost_basis_dict()
    stop_levels = state.stop_levels_dict()
    close_prices = _close_price_lookup(observation)
    preview = build_plan_impact_preview(
        config=config,
        observation=observation,
        current_batch=current_batch,
    )
    holdings_payload = [
        {
            "ticker": ticker,
            "shares": float(shares[ticker]),
            "average_cost": float(cost_basis.get(ticker, 0.0)),
            "market_value": float(market_values.get(ticker, 0.0)),
            "weight": (
                float(market_values.get(ticker, 0.0)) / float(state.total_nav)
                if state.total_nav > _EPSILON
                else 0.0
            ),
            "active_stop": float(stop_levels[ticker]) if ticker in stop_levels else None,
            "current_close": float(close_prices.get(ticker, 0.0)) if ticker in close_prices else None,
        }
        for ticker in sorted(shares)
    ]
    ticker_options = {
        action_type.value: _ticker_options_for_action(
            action_type=action_type,
            available_tickers=observation.available_tickers,
            holdings=shares,
            active_stop_tickers=stop_levels.keys(),
        )
        for action_type in (
            ActionType.BUY,
            ActionType.SELL,
            ActionType.REDUCE,
            ActionType.SET_STOP,
            ActionType.REMOVE_STOP,
        )
    }
    current_batch_payload = [
        {
            "action_type": action.action_type.value,
            "ticker": action.ticker,
            "quantity": float(action.quantity) if action.quantity is not None else None,
            "quantity_type": action.quantity_type.value if action.quantity_type is not None else None,
            "fraction": float(action.fraction) if action.fraction is not None else None,
            "stop_price": float(action.stop_price) if action.stop_price is not None else None,
            "summary": _action_summary(action),
            "detail": _action_detail(action),
        }
        for action in current_batch
    ]
    return {
        "current_week_index": int(observation.week_index),
        "current_date": observation.date.isoformat(),
        "max_actions_per_step": int(config.max_actions_per_step),
        "remaining_action_slots": max(0, config.max_actions_per_step - len(current_batch)),
        "available_tickers": list(observation.available_tickers),
        "current_cash": float(state.cash),
        "current_total_nav": float(state.total_nav),
        "current_batch": current_batch_payload,
        "holdings": holdings_payload,
        "active_stops": {ticker: float(value) for ticker, value in sorted(stop_levels.items())},
        "pending_liquidations": [
            {
                "ticker": item.ticker,
                "triggered_by_low": float(item.triggered_by_low),
                "stop_level": float(item.stop_level),
                "execution_week": int(item.execution_week),
            }
            for item in sorted(
                observation.pending_liquidations,
                key=lambda value: (value.execution_week, value.ticker),
            )
        ],
        "ticker_options": ticker_options,
        "close_prices": {ticker: float(value) for ticker, value in sorted(close_prices.items())},
        "plan_impact": {
            "estimated_spend": float(preview.estimated_spend),
            "estimated_proceeds": float(preview.estimated_proceeds),
            "estimated_transaction_costs": float(preview.estimated_transaction_costs),
            "estimated_remaining_cash": float(preview.estimated_remaining_cash),
            "estimated_positions_after": int(preview.estimated_positions_after),
            "estimated_invested_after": float(preview.estimated_invested_after),
            "projected_max_weight": preview.projected_max_weight,
            "warnings": list(preview.warnings),
            "notes": list(preview.notes),
        },
    }


def build_plan_impact_preview(
    *,
    config: SimulatorConfig,
    observation: Observation,
    current_batch: Sequence[Action],
) -> PlanImpactPreview:
    return _build_plan_impact_preview(config=config, observation=observation, current_batch=current_batch)


def _ticker_options_for_action(
    *,
    action_type: ActionType,
    available_tickers: Sequence[str],
    holdings: dict[str, float],
    active_stop_tickers: Sequence[str],
) -> list[str]:
    held_tickers = sorted(holdings.keys())
    stop_tickers = sorted(active_stop_tickers)
    if action_type == ActionType.BUY:
        return list(available_tickers)
    if action_type == ActionType.SELL:
        return held_tickers if held_tickers else list(available_tickers)
    if action_type in {ActionType.REDUCE, ActionType.SET_STOP}:
        return held_tickers
    if action_type == ActionType.REMOVE_STOP:
        return stop_tickers
    return list(available_tickers)


def _close_price_lookup(observation: Observation) -> dict[str, float]:
    return {str(row["ticker"]): float(row["close"]) for _, row in observation.current_week_ohlcv.iterrows()}


def _build_plan_impact_preview(
    *,
    config: SimulatorConfig,
    observation: Observation,
    current_batch: Sequence[Action],
) -> PlanImpactPreview:
    state = observation.portfolio_state
    close_prices = _close_price_lookup(observation)
    adv_lookup = _adv_shares_lookup(observation, config.adv_lookback_weeks)
    batch_start_nav = float(state.total_nav)
    projected_cash = float(state.cash)
    projected_shares = dict(state.shares_dict())
    stop_levels = dict(state.stop_levels_dict())
    estimated_spend = 0.0
    estimated_proceeds = 0.0
    estimated_costs = 0.0
    warnings: list[str] = []
    notes: list[str] = []
    ordered_batch = sorted(
        list(current_batch),
        key=lambda action: (_PREVIEW_ACTION_ORDER[action.action_type], _action_summary(action)),
    )
    if not ordered_batch:
        notes.append("No new decision is currently in the plan, so no new cash use is expected.")
    for action in ordered_batch:
        if action.action_type == ActionType.HOLD:
            notes.append("Choosing to do nothing this week does not change your cash or holdings.")
            continue
        ticker = action.ticker
        if ticker is None:
            warnings.append("One planned action is incomplete and may be rejected.")
            continue
        if action.action_type in {ActionType.SET_STOP, ActionType.REMOVE_STOP}:
            if action.action_type == ActionType.SET_STOP:
                stop_levels[ticker] = float(action.stop_price or 0.0)
            else:
                stop_levels.pop(ticker, None)
            continue
        reference_price = close_prices.get(ticker)
        if reference_price is None or reference_price <= 0.0:
            warnings.append(f"{ticker} cannot be estimated from visible prices.")
            continue
        current_shares = float(projected_shares.get(ticker, 0.0))
        signed_shares = _resolve_preview_shares(
            action=action,
            reference_price=float(reference_price),
            shares_held=current_shares,
            batch_start_nav=batch_start_nav,
        )
        if abs(signed_shares) <= _EPSILON:
            continue
        gross_trade_value = abs(signed_shares) * float(reference_price)
        trade_cost = _estimate_trade_cost(
            config=config,
            ticker=ticker,
            reference_price=float(reference_price),
            gross_trade_value=gross_trade_value,
            adv_lookup=adv_lookup,
        )
        estimated_costs += trade_cost
        if signed_shares > 0.0:
            estimated_spend += gross_trade_value
            projected_cash -= gross_trade_value + trade_cost
        else:
            estimated_proceeds += gross_trade_value
            projected_cash += gross_trade_value - trade_cost
        new_shares = max(0.0, current_shares + signed_shares)
        if new_shares <= _EPSILON:
            projected_shares.pop(ticker, None)
            stop_levels.pop(ticker, None)
        else:
            projected_shares[ticker] = float(new_shares)
    projected_market_values = {
        ticker: float(shares * close_prices[ticker])
        for ticker, shares in projected_shares.items()
        if shares > _EPSILON and ticker in close_prices
    }
    projected_nav = float(projected_cash + sum(projected_market_values.values()))
    estimated_invested_after = max(0.0, projected_nav - projected_cash)
    estimated_positions_after = sum(1 for shares in projected_shares.values() if shares > _EPSILON)
    projected_max_weight = None
    if projected_nav > _EPSILON and projected_market_values:
        projected_max_weight = max(projected_market_values.values()) / projected_nav
    return PlanImpactPreview(
        estimated_spend=float(estimated_spend),
        estimated_proceeds=float(estimated_proceeds),
        estimated_transaction_costs=float(estimated_costs),
        estimated_remaining_cash=float(projected_cash),
        estimated_positions_after=int(estimated_positions_after),
        estimated_invested_after=float(estimated_invested_after),
        projected_max_weight=projected_max_weight,
        warnings=tuple(dict.fromkeys(warnings)),
        notes=tuple(dict.fromkeys(notes)),
    )


def _resolve_preview_shares(
    *,
    action: Action,
    reference_price: float,
    shares_held: float,
    batch_start_nav: float,
) -> float:
    if action.action_type == ActionType.BUY:
        assert action.quantity_type is not None and action.quantity is not None
        if action.quantity_type == QuantityType.SHARES:
            return float(action.quantity)
        if action.quantity_type == QuantityType.NOTIONAL_DOLLARS:
            return float(action.quantity) / reference_price
        if action.quantity_type == QuantityType.NAV_FRACTION:
            return (float(action.quantity) * batch_start_nav) / reference_price
        return 0.0
    if action.action_type == ActionType.SELL:
        assert action.quantity_type is not None
        if action.quantity_type == QuantityType.CLOSE_ALL:
            return -float(shares_held)
        assert action.quantity is not None
        if action.quantity_type == QuantityType.SHARES:
            return -float(action.quantity)
        if action.quantity_type == QuantityType.NOTIONAL_DOLLARS:
            return -(float(action.quantity) / reference_price)
        return 0.0
    if action.action_type == ActionType.REDUCE:
        assert action.fraction is not None
        return -(float(shares_held) * float(action.fraction))
    return 0.0


def _adv_shares_lookup(observation: Observation, adv_lookback_weeks: int) -> dict[str, float]:
    history = observation.price_history.copy()
    if history.empty:
        return {}
    history = history.sort_values(["ticker", "_week_idx"])
    lookup: dict[str, float] = {}
    for ticker, group in history.groupby("ticker", sort=True):
        recent = group.tail(adv_lookback_weeks)
        if recent.empty:
            continue
        lookup[str(ticker)] = float(pd.to_numeric(recent["volume"], errors="coerce").mean())
    return lookup


def _estimate_trade_cost(
    *,
    config: SimulatorConfig,
    ticker: str,
    reference_price: float,
    gross_trade_value: float,
    adv_lookup: dict[str, float],
) -> float:
    adv_shares = float(adv_lookup.get(ticker, 0.0))
    adv_value = adv_shares * reference_price
    if adv_value <= 0.0:
        adv_value = max(gross_trade_value, reference_price)
    impact_bps = 0.0 if gross_trade_value <= 0.0 else (
        (gross_trade_value / adv_value) * config.impact_factor * 10_000.0
    )
    slippage_bps = config.base_slippage_bps + impact_bps
    return gross_trade_value * (
        config.commission_rate + config.spread_rate + (slippage_bps / 10_000.0)
    )


def _currency(value: float) -> str:
    return f"${value:,.2f}"


def _format_shares(value: float) -> str:
    rounded = round(float(value), 4)
    if rounded.is_integer():
        whole_shares = int(rounded)
        unit = "share" if whole_shares == 1 else "shares"
        return f"{whole_shares:,} {unit}"
    return f"{rounded:,.4f} shares"


def _action_summary(action: Action | None) -> str:
    if action is None:
        return "No action"
    if action.action_type == ActionType.HOLD:
        return "Do nothing this week"
    if action.action_type == ActionType.BUY:
        assert action.ticker and action.quantity_type and action.quantity is not None
        if action.quantity_type == QuantityType.SHARES:
            return f"Buy {_format_shares(action.quantity)} of {action.ticker}"
        if action.quantity_type == QuantityType.NOTIONAL_DOLLARS:
            return f"Buy {_currency(action.quantity)} of {action.ticker}"
        return f"Buy {action.quantity:.2%} of portfolio value in {action.ticker}"
    if action.action_type == ActionType.SELL:
        assert action.ticker and action.quantity_type
        if action.quantity_type == QuantityType.CLOSE_ALL:
            return f"Sell all shares of {action.ticker}"
        assert action.quantity is not None
        if action.quantity_type == QuantityType.SHARES:
            return f"Sell {_format_shares(action.quantity)} of {action.ticker}"
        return f"Sell {_currency(action.quantity)} of {action.ticker}"
    if action.action_type == ActionType.REDUCE:
        assert action.ticker and action.fraction is not None
        return f"Reduce {action.ticker} by {action.fraction:.0%}"
    if action.action_type == ActionType.SET_STOP:
        assert action.ticker and action.stop_price is not None
        return f"Set a stop price on {action.ticker} at {_currency(action.stop_price)}"
    if action.action_type == ActionType.REMOVE_STOP:
        assert action.ticker
        return f"Remove the stop price from {action.ticker}"
    return action.action_type.value.replace("_", " ").title()


def _action_detail(action: Action) -> str | None:
    if action.action_type == ActionType.BUY and action.quantity_type == QuantityType.NAV_FRACTION:
        return "This buy is sized as a share of your portfolio value at the start of the week."
    if action.action_type == ActionType.SELL and action.quantity_type == QuantityType.CLOSE_ALL:
        return "The simulator will try to fully close this holding."
    if action.action_type == ActionType.SET_STOP:
        return "If a later weekly low breaches this price, the simulator can schedule a forced sale."
    return None
