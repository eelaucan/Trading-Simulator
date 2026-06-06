"""Tech-focused dollar-cost averaging policy for Gemini AI sessions."""

from __future__ import annotations

from typing import Any

from simulator.actions import Action, ActionType, QuantityType
from simulator.config import SimulatorConfig
from simulator.observation import Observation

from .gemini_momentum import (
    _GEMINI_SINGLE_STOCK_CAP,
    _GEMINI_TURNOVER_CAP,
    create_gemini_simulator_config,
    is_gemini_simulator_config,
)

_EPSILON = 1e-6

# Mega-cap tech basket in DCA priority order (NVDA / Apple / Google first).
TECH_DCA_TICKERS: tuple[str, ...] = (
    "NVDA",
    "AAPL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "META",
)

_CASH_RESERVE = 0.02
_WEEKLY_BUY_BITE = 0.18


def tech_dca_universe(available_tickers: tuple[str, ...] | list[str]) -> list[str]:
    """Return configured tech tickers that exist in the active dataset."""
    available = set(available_tickers)
    return [ticker for ticker in TECH_DCA_TICKERS if ticker in available]


def tech_dca_target_weights(tickers: list[str], *, cash_reserve: float = _CASH_RESERVE) -> dict[str, float]:
    """Equal-weight targets across the tech basket."""
    if not tickers:
        return {}
    investable = max(0.0, 1.0 - cash_reserve)
    per_name = investable / float(len(tickers))
    return {ticker: float(per_name) for ticker in tickers}


def build_tech_dca_context(
    observation: Observation,
    simulator_config: SimulatorConfig,
) -> dict[str, Any]:
    """Visible signal payload for logs and prompts."""
    tickers = tech_dca_universe(observation.available_tickers)
    state = observation.portfolio_state
    nav = max(float(state.total_nav), 1e-12)
    cash_weight = float(state.cash) / nav
    current_weights = {
        ticker: float(state.market_value_dict().get(ticker, 0.0)) / nav
        for ticker in tickers
    }
    targets = tech_dca_target_weights(tickers)

    return {
        "strategy": "tech_dca",
        "cash_weight": cash_weight,
        "cash_reserve_target": _CASH_RESERVE,
        "max_positions": len(tickers),
        "tech_basket": tickers,
        "ranked_candidates": [
            {
                "rank": index + 1,
                "ticker": ticker,
                "score": None,
                "momentum_4w": None,
                "momentum_12w": None,
                "volatility": None,
                "latest_close": None,
                "weeks_observed": None,
                "current_weight": current_weights.get(ticker, 0.0),
                "target_weight": targets.get(ticker, 0.0),
            }
            for index, ticker in enumerate(tickers)
        ],
        "selected_focus_tickers": tickers,
        "suggested_target_weights": targets,
        "held_tickers": [
            ticker
            for ticker, weight in sorted(current_weights.items())
            if weight > 1e-6
        ],
    }


def apply_tech_dca_policy(
    observation: Observation,
    simulator_config: SimulatorConfig,
) -> list[Action]:
    """Buy mega-cap tech on a fixed weekly schedule; hold through drawdowns (no rotation sells)."""
    tickers = tech_dca_universe(observation.available_tickers)
    if not tickers:
        return [Action(action_type=ActionType.HOLD)]

    state = observation.portfolio_state
    nav = max(float(state.total_nav), 1e-12)
    market_values = state.market_value_dict()
    current_weights = {
        ticker: float(market_values.get(ticker, 0.0)) / nav for ticker in tickers
    }
    targets = tech_dca_target_weights(tickers)

    projected_cash_weight = float(state.cash) / nav
    remaining_turnover = float(simulator_config.turnover_cap)
    stock_cap = min(
        float(simulator_config.single_stock_cap),
        _GEMINI_SINGLE_STOCK_CAP,
    ) - 0.01
    actions: list[Action] = []

    for ticker in tickers:
        target_weight = targets[ticker]
        current_weight = current_weights.get(ticker, 0.0)
        if current_weight >= target_weight - 0.005:
            continue
        if current_weight >= stock_cap - 0.005:
            continue

        available_cash_weight = projected_cash_weight - _CASH_RESERVE
        desired_buy_weight = min(target_weight, stock_cap) - current_weight
        buy_weight = min(
            desired_buy_weight,
            _WEEKLY_BUY_BITE,
            available_cash_weight,
            remaining_turnover,
        )
        if buy_weight <= _EPSILON:
            continue

        actions.append(
            Action(
                action_type=ActionType.BUY,
                ticker=ticker,
                quantity=float(buy_weight),
                quantity_type=QuantityType.NAV_FRACTION,
            )
        )
        projected_cash_weight = max(0.0, projected_cash_weight - buy_weight)
        remaining_turnover = max(0.0, remaining_turnover - buy_weight)
        if len(actions) >= int(simulator_config.max_actions_per_step):
            break

    if not actions:
        return [Action(action_type=ActionType.HOLD)]
    return actions


def tech_dca_rationale(signal_context: dict[str, Any], actions: list[Action]) -> str:
    """Human-readable summary for decision logs."""
    basket = signal_context.get("tech_basket") or TECH_DCA_TICKERS
    labels = ", ".join(basket[:3])
    if len(basket) > 3:
        labels = f"{labels}, and peers"

    if len(actions) == 1 and actions[0].action_type == ActionType.HOLD:
        return (
            f"Tech DCA targets met this week. Holding {labels} through volatility — "
            "no panic selling."
        )

    bought = [
        action.ticker
        for action in actions
        if action.action_type == ActionType.BUY and action.ticker
    ]
    if bought:
        names = ", ".join(dict.fromkeys(bought))
        return (
            f"DCA buy into {names}. Steady weekly adds into mega-cap tech "
            f"({labels}); hold positions even when prices dip."
        )
    return "Maintain tech DCA allocations."
