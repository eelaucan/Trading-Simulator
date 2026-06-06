import type { TradePlannerEventPayload, TradePlannerProps } from "../types.js";

export interface DatasetOption {
  id: string;
  label: string;
  path: string;
}

export interface SessionMetadata {
  participant_id: string;
  condition: string;
  condition_label: string;
  episode_name: string;
  dataset_path: string;
  started_at: string;
  finished_at: string | null;
  decision_start_week: number;
  visible_history_weeks_at_start: number;
  notes: string | null;
}

export interface PortfolioPayload {
  cash: number;
  total_nav: number;
  invested: number;
  positions: number;
  weekly_turnover: number;
  concentration_hhi: number;
  portfolio_volatility: number | null;
  nav_history: number[];
  holdings: Array<{
    ticker: string;
    shares: number;
    average_cost: number;
    market_value: number;
    weight: number;
    active_stop: number | null;
  }>;
  allocation: Array<{ label: string; value: number; weight: number }>;
}

export interface ObservationPayload {
  week_index: number;
  date: string;
  available_tickers: string[];
  pending_liquidations: Array<{
    ticker: string;
    triggered_by_low: number;
    stop_level: number;
    execution_week: number;
  }>;
  market_rows: Array<{
    ticker: string;
    close: number;
    open: number;
    low: number;
    high: number;
    volume: number;
    change_vs_previous_close: number | null;
  }>;
  price_history: Array<{ date: string; ticker: string; close: number }>;
}

export interface MetricsPayload {
  total_return: number;
  max_drawdown: number;
  realized_vol: number | null;
  sharpe_ratio: number | null;
  avg_hhi: number;
  avg_weekly_turnover: number;
  blow_up_flag: boolean;
  n_invalid_attempts: number;
  n_clipped_trades: number;
  n_stop_triggers: number;
  n_gap_adjustments: number;
  vol_rule_activation_week: number | null;
}

export interface SessionResponse {
  session: string;
  status: "not_started" | "running" | "finished";
  run_mode: string;
  metadata: SessionMetadata;
  portfolio: PortfolioPayload;
  observation?: ObservationPayload;
  planner_props?: TradePlannerProps;
  metrics?: MetricsPayload;
  last_step_info?: Record<string, unknown>;
  error?: string | null;
  done: boolean;
  llm_decision_log?: Array<Record<string, unknown>>;
  gemini_summary?: {
    decisions: number;
    fallback_weeks: number;
    trade_weeks: number;
    last_error: string | null;
  };
}

export interface StartSessionInput {
  participant_id: string;
  condition: string;
  run_mode: string;
  episode_name: string;
  dataset_path: string;
  notes: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.text();
    let message = detail || `Request failed (${response.status})`;
    try {
      const parsed = JSON.parse(detail) as { detail?: string };
      if (typeof parsed.detail === "string") {
        message = parsed.detail;
      }
    } catch {
      // Keep raw response text when the API does not return JSON.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export async function fetchDatasets(): Promise<DatasetOption[]> {
  const payload = await request<{ datasets: DatasetOption[] }>("/api/datasets");
  return payload.datasets;
}

export async function startSession(input: StartSessionInput): Promise<SessionResponse> {
  return request<SessionResponse>("/api/session/start", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function advanceGeminiStep(session: string): Promise<SessionResponse> {
  return request<SessionResponse>("/api/session/ai-step", {
    method: "POST",
    body: JSON.stringify({ session }),
  });
}

export async function sendPlannerEvent(
  session: string,
  event: TradePlannerEventPayload | null,
): Promise<SessionResponse> {
  return request<SessionResponse>("/api/session/planner", {
    method: "POST",
    body: JSON.stringify({ session, event }),
  });
}
