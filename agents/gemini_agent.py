"""Gemini-backed autonomous trading agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from typing import Any, Protocol

from simulator.actions import Action, ActionType
from simulator.config import SimulatorConfig
from simulator.observation import Observation

from .llm_actions import actions_from_llm_payload, extract_json_object
from .llm_policy import apply_signal_rescue, should_apply_signal_rescue
from .llm_prompt import build_trading_prompt
from .llm_signals import build_signal_context


_DEPRECATED_GEMINI_MODELS: dict[str, str] = {
    "gemini-2.0-flash": "gemini-2.5-flash-lite",
    "gemini-2.0-flash-001": "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite-001": "gemini-2.5-flash-lite",
    "gemini-3.5-flash": "gemini-2.5-flash-lite",
    "gemini-3-flash-preview": "gemini-2.5-flash-lite",
}


_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
_DEFAULT_GEMINI_TIMEOUT_SECONDS = 8.0


def resolve_gemini_model(requested: str | None) -> str:
    """Map retired Gemini model IDs to currently supported replacements."""
    normalized = (requested or _DEFAULT_GEMINI_MODEL).strip() or _DEFAULT_GEMINI_MODEL
    return _DEPRECATED_GEMINI_MODELS.get(normalized, normalized)


class GeminiClient(Protocol):
    def generate_json(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from the model."""


@dataclass(frozen=True, slots=True)
class GeminiAgentConfig:
    model: str = _DEFAULT_GEMINI_MODEL
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
    decision_source: str = "llm"
    selected_focus: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["date"] = self.date.isoformat()
        payload["selected_focus"] = list(self.selected_focus)
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

    def generate_json(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        generation_config = self._genai.GenerationConfig(
            temperature=float(temperature),
            response_mime_type="application/json",
        )
        gemini_model = self._genai.GenerativeModel(model)
        timeout_seconds = _gemini_timeout_seconds()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                gemini_model.generate_content,
                prompt,
                generation_config=generation_config,
            )
            try:
                response = future.result(timeout=timeout_seconds)
            except FuturesTimeoutError as exc:
                raise TimeoutError(
                    f"Gemini request timed out after {timeout_seconds:.0f}s."
                ) from exc
        text = getattr(response, "text", None) or ""
        if not text.strip():
            block_reason = _response_block_reason(response)
            if block_reason:
                raise ValueError(block_reason)
            raise ValueError("Model response was empty.")
        return extract_json_object(text)


def _response_block_reason(response: Any) -> str | None:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            return f"Prompt blocked by Gemini safety filters: {block_reason}"

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason and str(finish_reason).upper() not in {"STOP", "FINISHREASON.STOP"}:
            return f"Gemini stopped generation: {finish_reason}"
    return None


def _gemini_timeout_seconds() -> float:
    raw = os.environ.get("GEMINI_TIMEOUT_SECONDS", str(_DEFAULT_GEMINI_TIMEOUT_SECONDS))
    try:
        return max(3.0, min(float(raw), 25.0))
    except ValueError:
        return _DEFAULT_GEMINI_TIMEOUT_SECONDS


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
        signal_context = build_signal_context(observation, self.simulator_config)
        prompt = build_trading_prompt(observation, self.simulator_config)
        client = self._client or build_gemini_client()
        raw_response = ""
        model_name = resolve_gemini_model(os.environ.get("GEMINI_MODEL", self.config.model))
        try:
            payload = client.generate_json(
                prompt=prompt,
                model=model_name,
                temperature=self.config.temperature,
            )
            raw_response = json.dumps(payload, sort_keys=True)
            rationale, actions = actions_from_llm_payload(
                payload,
                self.simulator_config.max_actions_per_step,
            )
            decision_source = "llm"
            if should_apply_signal_rescue(observation, actions, signal_context):
                actions = apply_signal_rescue(observation, self.simulator_config)
                rationale = (
                    f"{rationale} Signal rescue applied after under-investment.".strip()
                    if rationale
                    else "Signal rescue applied after under-investment."
                )
                decision_source = "signal_rescue"
            self._record_decision(
                observation=observation,
                rationale=rationale,
                actions=actions,
                raw_response=raw_response,
                model_name=model_name,
                used_fallback=False,
                decision_source=decision_source,
                signal_context=signal_context,
            )
            return actions
        except Exception as exc:
            if should_apply_signal_rescue(
                observation,
                [Action(action_type=ActionType.HOLD)],
                signal_context,
            ):
                actions = apply_signal_rescue(observation, self.simulator_config)
                self._record_decision(
                    observation=observation,
                    rationale="Signal rescue applied after model failure.",
                    actions=actions,
                    raw_response=raw_response,
                    model_name=model_name,
                    used_fallback=False,
                    decision_source="signal_rescue",
                    signal_context=signal_context,
                    error=str(exc),
                )
                return actions
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
                decision_source="fallback_hold",
                signal_context=signal_context,
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
        decision_source: str,
        signal_context: dict[str, Any] | None = None,
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
        selected_focus = tuple(signal_context.get("selected_focus_tickers", [])) if signal_context else ()
        self._decision_records.append(
            GeminiDecisionRecord(
                week_index=int(observation.week_index),
                date=observation.date,
                rationale=rationale,
                model=model_name,
                raw_response=raw_response,
                generated_actions=tuple(generated_actions),
                used_fallback=used_fallback,
                decision_source=decision_source,
                selected_focus=selected_focus,
                error=error,
            )
        )
