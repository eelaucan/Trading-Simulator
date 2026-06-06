"""FastAPI entrypoint for the Vercel-hosted trading simulator."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import advance_gemini_session, discover_datasets, handle_planner_event, start_session


app = FastAPI(title="Trading Simulator API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartSessionRequest(BaseModel):
    participant_id: str = ""
    condition: str = "human_only"
    run_mode: str = "human"
    episode_name: str = "pilot_episode_01"
    dataset_path: str = "data/sample/weekly_ohlcv_synthetic.csv"
    notes: str = ""


class PlannerEventRequest(BaseModel):
    session: str
    event: dict[str, Any] | None = None


class GeminiStepRequest(BaseModel):
    session: str


@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    return {"datasets": discover_datasets()}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session/start")
def create_session(request: StartSessionRequest) -> dict[str, Any]:
    try:
        return start_session(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/session/ai-step")
def gemini_step(request: GeminiStepRequest) -> dict[str, Any]:
    try:
        return advance_gemini_session(request.session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/session/planner")
def planner_event(request: PlannerEventRequest) -> dict[str, Any]:
    try:
        return handle_planner_event(request.session, request.event)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(exc)) from exc
