"""Session orchestration for the web API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any

from agents.gemini_momentum import create_gemini_simulator_config
from agents.gemini_tech_dca import apply_tech_dca_policy, tech_dca_rationale
from agents.llm_signals import build_signal_context
from agents.runner import run_benchmark_agent
from simulator.actions import Action, ActionType
from simulator.actions import Action
from simulator.config import SimulatorConfig
from simulator.env import TradingEnvironment
from simulator.market import MarketReplay
from simulator.metrics import SimulationMetrics
from simulator.observation import Observation
from simulator.state import PortfolioState
from ui.session import SessionMetadata, SessionStatus

from .serializers import actions_from_payload, session_payload
from .session_codec import decode_runtime, encode_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "sample" / "weekly_ohlcv_synthetic.csv"


@dataclass(slots=True)
class RuntimeSession:
    env: TradingEnvironment
    metadata: SessionMetadata
    status: SessionStatus
    run_mode: str
    observation: Observation | None
    state: PortfolioState
    current_batch: list[Action]
    metrics: SimulationMetrics | None = None
    last_step_info: dict[str, Any] | None = None
    llm_decision_log: list[dict[str, Any]] | None = None
    error: str | None = None


def discover_datasets() -> list[dict[str, str]]:
    candidates = [
        DEFAULT_DATASET,
        PROJECT_ROOT / "thesis" / "data" / "sample" / "weekly_ohlcv_synthetic.csv",
    ]
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        key = str(path.relative_to(PROJECT_ROOT))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": key, "label": path.name, "path": key})
    if not rows and DEFAULT_DATASET.exists():
        rows.append(
            {
                "id": str(DEFAULT_DATASET.relative_to(PROJECT_ROOT)),
                "label": DEFAULT_DATASET.name,
                "path": str(DEFAULT_DATASET.relative_to(PROJECT_ROOT)),
            }
        )
    return rows


def _resolve_dataset_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise ValueError(f"Dataset not found: {raw_path}")
    return candidate


def _visible_history_weeks(observation: Observation) -> int:
    if observation.price_history.empty:
        return 0
    return int(observation.price_history["date"].nunique())


def _ai_benchmark_output_dir() -> Path | None:
    """Writable export directory for local runs; None on read-only hosts like Vercel."""
    if os.environ.get("VERCEL"):
        return None
    preferred = PROJECT_ROOT / "output" / "ai_benchmark"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        return None
    return preferred


def start_session(payload: dict[str, Any]) -> dict[str, Any]:
    run_mode = str(payload.get("run_mode", "human")).strip().lower()
    if run_mode == "ai_benchmark":
        return _start_ai_session(payload)
    if run_mode == "ai_gemini":
        return _start_gemini_session(payload)
    return _start_human_session(payload)


def _start_human_session(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_path = _resolve_dataset_path(str(payload.get("dataset_path", "")).strip())
    market = MarketReplay(str(dataset_path))
    config = SimulatorConfig(ticker_universe=market.available_tickers)
    env = TradingEnvironment(market=market, config=config)
    observation, state = env.reset()
    started_at = datetime.now().astimezone()
    participant_id = str(payload.get("participant_id", "")).strip() or "participant_01"
    metadata = SessionMetadata(
        participant_id=participant_id,
        condition=str(payload.get("condition", "human_only")).strip(),
        episode_name=str(payload.get("episode_name", "pilot_episode_01")).strip(),
        dataset_path=str(dataset_path.relative_to(PROJECT_ROOT)),
        started_at=started_at,
        decision_start_week=env.initial_decision_week,
        visible_history_weeks_at_start=_visible_history_weeks(observation),
        notes=(str(payload.get("notes", "")).strip() or None),
    )
    status = SessionStatus.FINISHED if env.done else SessionStatus.RUNNING
    metrics = env.get_metrics() if status == SessionStatus.FINISHED else None
    if status == SessionStatus.FINISHED:
        metadata = metadata.mark_finished(started_at)

    runtime = RuntimeSession(
        env=env,
        metadata=metadata,
        status=status,
        run_mode="human",
        observation=observation if status == SessionStatus.RUNNING else None,
        state=state,
        current_batch=[],
        metrics=metrics,
    )
    return _response(runtime)


def _start_ai_session(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_path = _resolve_dataset_path(str(payload.get("dataset_path", "")).strip())
    started_at = datetime.now().astimezone()
    participant_id = str(payload.get("participant_id", "benchmark_agent")).strip() or "benchmark_agent"
    episode_name = str(payload.get("episode_name", "pilot_episode_01")).strip() or "pilot_episode_01"
    result = run_benchmark_agent(
        data_path=str(dataset_path),
        output_dir=_ai_benchmark_output_dir(),
        output_prefix=f"{participant_id}_{episode_name}",
    )
    metadata = SessionMetadata(
        participant_id=participant_id,
        condition="ai_benchmark",
        episode_name=episode_name,
        dataset_path=str(dataset_path.relative_to(PROJECT_ROOT)),
        started_at=started_at,
        decision_start_week=result.env.initial_decision_week,
        visible_history_weeks_at_start=_visible_history_weeks(result.initial_observation),
        notes=(str(payload.get("notes", "")).strip() or None),
    ).mark_finished(datetime.now().astimezone())

    runtime = RuntimeSession(
        env=result.env,
        metadata=metadata,
        status=SessionStatus.FINISHED,
        run_mode="ai_benchmark",
        observation=None,
        state=result.final_state,
        current_batch=[],
        metrics=result.metrics,
    )
    response = _response(runtime)
    if result.output_paths:
        response["ai_export_paths"] = {
            label: str(path) for label, path in result.output_paths.items()
        }
    return response


def _start_gemini_session(payload: dict[str, Any]) -> dict[str, Any]:
    """Start an interactive Gemini session that advances one decision week per API call."""
    _require_gemini_api_key()
    dataset_path = _resolve_dataset_path(str(payload.get("dataset_path", "")).strip())
    market = MarketReplay(str(dataset_path))
    config = create_gemini_simulator_config(market.available_tickers)
    env = TradingEnvironment(market=market, config=config)
    observation, state = env.reset()
    started_at = datetime.now().astimezone()
    participant_id = str(payload.get("participant_id", "gemini_agent")).strip() or "gemini_agent"
    metadata = SessionMetadata(
        participant_id=participant_id,
        condition="ai_gemini",
        episode_name=str(payload.get("episode_name", "pilot_episode_01")).strip(),
        dataset_path=str(dataset_path.relative_to(PROJECT_ROOT)),
        started_at=started_at,
        decision_start_week=env.initial_decision_week,
        visible_history_weeks_at_start=_visible_history_weeks(observation),
        notes=(str(payload.get("notes", "")).strip() or None),
    )
    status = SessionStatus.FINISHED if env.done else SessionStatus.RUNNING
    metrics = env.get_metrics() if status == SessionStatus.FINISHED else None
    if status == SessionStatus.FINISHED:
        metadata = metadata.mark_finished(started_at)

    runtime = RuntimeSession(
        env=env,
        metadata=metadata,
        status=status,
        run_mode="ai_gemini",
        observation=observation if status == SessionStatus.RUNNING else None,
        state=state,
        current_batch=[],
        metrics=metrics,
        llm_decision_log=[],
    )
    return _response(runtime)


def advance_gemini_session(token: str) -> dict[str, Any]:
    """Advance one Gemini decision week."""
    return advance_gemini_session_batch(token, max_steps=1)


def advance_gemini_session_batch(
    token: str,
    *,
    max_steps: int = 1,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Advance up to ``max_steps`` weeks within a serverless time budget."""
    _require_gemini_api_key()
    runtime = decode_runtime(token)
    if runtime.run_mode != "ai_gemini":
        raise ValueError("Session is not a Gemini AI session.")
    if runtime.status != SessionStatus.RUNNING or runtime.observation is None:
        raise ValueError("Session is not active.")

    steps = max(1, min(int(max_steps), 5))
    budget = time_budget_seconds if time_budget_seconds is not None else 8.5
    deadline = time.monotonic() + max(3.0, min(float(budget), 25.0))

    response: dict[str, Any] | None = None
    completed = 0
    while (
        runtime.status == SessionStatus.RUNNING
        and runtime.observation is not None
        and completed < steps
        and time.monotonic() < deadline
    ):
        _run_one_gemini_step(runtime)
        completed += 1
        response = _response(runtime)

    if response is None:
        raise ValueError("Session is not active.")
    response["batch_steps"] = completed
    return response


def _run_one_gemini_step(runtime: RuntimeSession) -> None:
    observation = runtime.observation
    if observation is None:
        raise ValueError("Session is not active.")

    signal_context = build_signal_context(observation, runtime.env.config)
    model_name = resolve_gemini_model(os.environ.get("GEMINI_MODEL"))
    actions = apply_tech_dca_policy(observation, runtime.env.config)
    decision_source = "tech_dca"
    rationale = tech_dca_rationale(signal_context, actions)
    error: str | None = None

    decision_record = {
        "week_index": int(observation.week_index),
        "date": observation.date.isoformat(),
        "rationale": rationale,
        "model": model_name,
        "generated_actions": _actions_to_dicts(actions),
        "used_fallback": decision_source == "fallback_hold",
        "decision_source": decision_source,
        "selected_focus": list(signal_context.get("selected_focus_tickers", [])),
        "error": error,
    }

    runtime.current_batch = actions
    if runtime.llm_decision_log is None:
        runtime.llm_decision_log = []
    runtime.llm_decision_log.append(decision_record)
    runtime.error = None
    _submit_batch(runtime)


def _compact_decision_record(record: dict[str, Any]) -> dict[str, Any]:
    compact = dict(record)
    compact.pop("raw_response", None)
    return compact


def _actions_to_dicts(actions: list[Action]) -> list[dict[str, Any]]:
    return [
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


def resolve_gemini_model(requested: str | None) -> str:
    from agents.gemini_agent import resolve_gemini_model as _resolve

    return _resolve(requested)


def _require_gemini_api_key() -> None:
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise ValueError(
            "GEMINI_API_KEY is not configured. Add it to your Vercel project environment variables."
        )


def handle_planner_event(token: str, event: dict[str, Any] | None) -> dict[str, Any]:
    runtime = decode_runtime(token)
    if runtime.status != SessionStatus.RUNNING or runtime.observation is None:
        raise ValueError("Session is not active.")

    if not event:
        return _response(runtime)

    try:
        next_batch = actions_from_payload(
            event.get("actions"),
            runtime.env.config.max_actions_per_step,
        )
    except ValueError as exc:
        runtime.error = f"Planner error: {exc}"
        return _response(runtime)

    runtime.current_batch = next_batch
    runtime.error = None
    event_type = str(event.get("event_type", "plan_change")).strip().lower()
    if event_type == "submit":
        return _submit_batch(runtime)
    return _response(runtime)


def _submit_batch(runtime: RuntimeSession) -> dict[str, Any]:
    env = runtime.env
    observation = runtime.observation
    if observation is None:
        raise ValueError("Session is not active.")

    previous_state = runtime.state
    try:
        next_observation, next_state, done, info = env.step(runtime.current_batch)
    except Exception as exc:
        runtime.error = f"Step failed: {exc}"
        return _response(runtime)

    runtime.error = None
    runtime.observation = next_observation
    runtime.state = next_state
    runtime.last_step_info = _augment_step_info(
        previous_state=previous_state,
        next_state=next_state,
        info=info,
    )
    runtime.current_batch = []

    if done:
        runtime.metadata = runtime.metadata.mark_finished(datetime.now().astimezone())
        runtime.status = SessionStatus.FINISHED
        runtime.metrics = env.get_metrics()
        runtime.observation = None
    return _response(runtime)


def _response(runtime: RuntimeSession) -> dict[str, Any]:
    include_planner = runtime.run_mode != "ai_gemini"
    body = session_payload(
        env=runtime.env,
        metadata=runtime.metadata,
        status=runtime.status.value,
        observation=runtime.observation,
        state=runtime.state,
        current_batch=runtime.current_batch,
        metrics=runtime.metrics,
        last_step_info=runtime.last_step_info,
        error=runtime.error,
        run_mode=runtime.run_mode,
        include_planner=include_planner,
    )
    body["session"] = encode_runtime(runtime)
    if runtime.llm_decision_log:
        if runtime.status == SessionStatus.FINISHED:
            body["llm_decision_log"] = list(runtime.llm_decision_log)
        else:
            body["latest_decision"] = runtime.llm_decision_log[-1]
        fallback_weeks = sum(
            1 for record in runtime.llm_decision_log if record.get("used_fallback")
        )
        last_error = next(
            (
                str(record.get("error"))
                for record in reversed(runtime.llm_decision_log)
                if record.get("error")
            ),
            None,
        )
        strategy_weeks = sum(
            1
            for record in runtime.llm_decision_log
            if record.get("decision_source") in {"signal_rescue", "momentum_agent", "tech_dca"}
        )
        body["gemini_summary"] = {
            "decisions": len(runtime.llm_decision_log),
            "fallback_weeks": fallback_weeks,
            "trade_weeks": len(runtime.llm_decision_log) - fallback_weeks,
            "signal_rescue_weeks": strategy_weeks,
            "last_error": last_error,
        }
    return body


def _augment_step_info(
    *,
    previous_state: PortfolioState,
    next_state: PortfolioState,
    info: dict[str, object],
) -> dict[str, object]:
    previous_shares = previous_state.shares_dict()
    next_shares = next_state.shares_dict()
    position_change_items: list[str] = []
    for ticker in sorted(set(previous_shares) | set(next_shares)):
        before_shares = float(previous_shares.get(ticker, 0.0))
        after_shares = float(next_shares.get(ticker, 0.0))
        if before_shares <= 1e-12 and after_shares > 1e-12:
            position_change_items.append(f"{ticker}: opened a new position.")
        elif before_shares > 1e-12 and after_shares <= 1e-12:
            position_change_items.append(f"{ticker}: fully removed from the portfolio.")
        elif after_shares > before_shares + 1e-12:
            position_change_items.append(f"{ticker}: position increased.")
        elif after_shares < before_shares - 1e-12:
            position_change_items.append(f"{ticker}: position reduced.")
    augmented = dict(info)
    augmented.update(
        {
            "cash_before": float(previous_state.cash),
            "cash_after": float(next_state.cash),
            "total_nav_before": float(previous_state.total_nav),
            "total_nav_after": float(next_state.total_nav),
            "position_change_items": position_change_items,
        }
    )
    return augmented
