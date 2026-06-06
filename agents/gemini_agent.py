"""Gemini-backed autonomous trading agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from typing import Any, Protocol

from simulator.actions import Action, ActionType
from simulator.config import SimulatorConfig
from simulator.observation import Observation

from .llm_actions import actions_from_llm_payload, extract_json_object
from .llm_prompt import build_trading_prompt


class GeminiClient(Protocol):
    def generate_json(self, *, prompt: str, model: str) -> dict[str, Any]:
        """Return a parsed JSON object from the model."""


@dataclass(frozen=True, slots=True)
class GeminiAgentConfig:
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2
    fallback_to_hold: bool = True


@dataclass(frozen=True, slots=True)
class GeminiDecisionRecord:
    week_index: int
    date: datetime
    rationale: str
    model: str
    raw_response: str
    generated_actions: tuple[dict[str, Any], ...]
    used_fallback: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["date"] = self.date.isoformat()
        return payload


class HttpGeminiClient:
    """Thin wrapper around the Google Generative AI SDK."""

    def __init__(self, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY is not configured.")
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "google-generativeai is not installed. Add it to requirements.txt."
            ) from exc
        genai.configure(api_key=api_key.strip())
        self._genai = genai

    def generate_json(self, *, prompt: str, model: str) -> dict[str, Any]:
        generation_config = self._genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
        )
        gemini_model = self._genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt, generation_config=generation_config)
        text = getattr(response, "text", None) or ""
        return extract_json_object(text)


def build_gemini_client() -> GeminiClient:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return HttpGeminiClient(api_key)


class GeminiTradingAgent:
    """Weekly decision agent backed by Gemini."""

    def __init__(
        self,
        *,
        simulator_config: SimulatorConfig,
        config: GeminiAgentConfig | None = None,
        client: GeminiClient | None = None,
    ) -> None:
        self.simulator_config = simulator_config
        self.config = config or GeminiAgentConfig()
        self._client = client
        self._decision_records: list[GeminiDecisionRecord] = []

    @property
    def decision_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(record.to_dict() for record in self._decision_records)

    def reset_log(self) -> None:
        self._decision_records.clear()

    def decide(self, observation: Observation) -> list[Action]:
        prompt = build_trading_prompt(observation, self.simulator_config)
        client = self._client or build_gemini_client()
        raw_response = ""
        model_name = os.environ.get("GEMINI_MODEL", self.config.model).strip() or self.config.model
        try:
            payload = client.generate_json(prompt=prompt, model=model_name)
            raw_response = json.dumps(payload, sort_keys=True)
            rationale, actions = actions_from_llm_payload(
                payload,
                self.simulator_config.max_actions_per_step,
            )
            self._record_decision(
                observation=observation,
                rationale=rationale,
                actions=actions,
                raw_response=raw_response,
                model_name=model_name,
                used_fallback=False,
            )
            return actions
        except Exception as exc:
            if not self.config.fallback_to_hold:
                raise
            hold = [Action(action_type=ActionType.HOLD)]
            self._record_decision(
                observation=observation,
                rationale="",
                actions=hold,
                raw_response=raw_response,
                model_name=model_name,
                used_fallback=True,
                error=str(exc),
            )
            return hold

    def _record_decision(
        self,
        *,
        observation: Observation,
        rationale: str,
        actions: list[Action],
        raw_response: str,
        model_name: str,
        used_fallback: bool,
        error: str | None = None,
    ) -> None:
        generated_actions = [
            {
                "action_type": action.action_type.value,
                "ticker": action.ticker,
                "quantity": action.quantity,
                "quantity_type": (
                    None if action.quantity_type is None else action.quantity_type.value
                ),
                "fraction": action.fraction,
                "stop_price": action.stop_price,
            }
            for action in actions
        ]
        self._decision_records.append(
            GeminiDecisionRecord(
                week_index=int(observation.week_index),
                date=observation.date,
                rationale=rationale,
                model=model_name,
                raw_response=raw_response,
                generated_actions=tuple(generated_actions),
                used_fallback=used_fallback,
                error=error,
            )
        )
