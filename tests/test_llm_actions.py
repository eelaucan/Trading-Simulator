"""Tests for LLM action parsing helpers."""

from __future__ import annotations

import pytest

from agents.llm_actions import actions_from_llm_payload, extract_json_object
from simulator.actions import ActionType, QuantityType


def test_extract_json_object_from_fenced_response() -> None:
    payload = extract_json_object(
        '```json\n{"rationale": "stay defensive", "actions": [{"action_type": "hold"}]}\n```'
    )
    assert payload["rationale"] == "stay defensive"


def test_actions_from_llm_payload_parses_buy_action() -> None:
    rationale, actions = actions_from_llm_payload(
        {
            "rationale": "add exposure",
            "actions": [
                {
                    "action_type": "buy",
                    "ticker": "AAPL",
                    "quantity": 0.1,
                    "quantity_type": "nav_fraction",
                }
            ],
        },
        max_actions_per_step=5,
    )
    assert rationale == "add exposure"
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.BUY
    assert actions[0].ticker == "AAPL"
    assert actions[0].quantity_type == QuantityType.NAV_FRACTION


def test_actions_from_llm_payload_rejects_hold_mixed_with_trade() -> None:
    with pytest.raises(ValueError, match="hold week cannot include other actions"):
        actions_from_llm_payload(
            {
                "actions": [
                    {"action_type": "buy", "ticker": "AAPL", "quantity": 1, "quantity_type": "shares"},
                    {"action_type": "hold"},
                ]
            },
            max_actions_per_step=5,
        )
