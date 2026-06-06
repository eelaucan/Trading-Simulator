"""Parse LLM JSON responses into simulator actions."""

from __future__ import annotations

import json
import re
from typing import Any

from simulator.actions import Action, ActionType, QuantityType


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object found in a model response."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model response was empty.")

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Model response did not contain a JSON object.")


def actions_from_llm_payload(payload: object, max_actions_per_step: int) -> tuple[str, list[Action]]:
    """Convert an LLM JSON payload into validated simulator actions."""
    if not isinstance(payload, dict):
        raise ValueError("LLM payload must be a JSON object.")

    rationale = str(payload.get("rationale", "")).strip()
    raw_actions = payload.get("actions")
    if raw_actions is None:
        raise ValueError("LLM payload must include an actions array.")
    if not isinstance(raw_actions, list):
        raise ValueError("LLM actions must be an array.")

    batch: list[Action] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            raise ValueError("Each LLM action must be an object.")
        action = _action_from_dict(item)
        cleaned = [entry for entry in batch if entry.action_type != ActionType.HOLD]
        if len(cleaned) >= max_actions_per_step:
            raise ValueError("LLM returned more than the maximum number of actions.")
        if action.action_type == ActionType.HOLD and cleaned:
            raise ValueError("A hold week cannot include other actions.")
        batch = list(cleaned)
        batch.append(action)

    if not batch:
        batch = [Action(action_type=ActionType.HOLD)]
    return rationale, batch


def _action_from_dict(payload: dict[str, Any]) -> Action:
    action_type_raw = payload.get("action_type")
    if not isinstance(action_type_raw, str):
        raise ValueError("Each action must include action_type.")
    action_type = ActionType(action_type_raw.strip().lower())

    ticker_value = payload.get("ticker")
    ticker = ticker_value.strip() if isinstance(ticker_value, str) and ticker_value.strip() else None

    quantity_type_raw = payload.get("quantity_type")
    quantity_type = (
        None
        if quantity_type_raw in (None, "")
        else QuantityType(str(quantity_type_raw).strip().lower())
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
