import {
  fetchDatasets,
  sendPlannerEvent,
  startSession,
  type DatasetOption,
  type SessionResponse,
} from "./api/client.js";
import {
  dataTable,
  drawdownChart,
  holdingsPanel,
  insightChips,
  lineChart,
  marketPanel,
  metricGrid,
  noteCard,
  pendingLiquidationsPanel,
  portfolioInsightChips,
  sectionHeader,
} from "./components/ui.js";
import { TradePlannerApp } from "./TradePlanner.js";
import type { TradePlannerEventPayload, TradePlannerProps } from "./types.js";
import { currency, pct } from "./utils/format.js";

type View = "setup" | "session" | "finished";

export class TradingSimulatorApp {
  private readonly root: HTMLElement;
  private view: View = "setup";
  private datasets: DatasetOption[] = [];
  private session: SessionResponse | null = null;
  private selectedTicker = "";
  private planner: TradePlannerApp | null = null;
  private plannerHost: HTMLElement | null = null;
  private loading = false;

  constructor(root: HTMLElement) {
    this.root = root;
  }

  public async init(): Promise<void> {
    try {
      this.datasets = await fetchDatasets();
    } catch {
      this.datasets = [
        {
          id: "data/sample/weekly_ohlcv_synthetic.csv",
          label: "weekly_ohlcv_synthetic.csv",
          path: "data/sample/weekly_ohlcv_synthetic.csv",
        },
      ];
    }
    this.render();
  }

  private render(): void {
    if (this.view === "setup") {
      this.renderSetup();
      return;
    }
    if (!this.session) {
      this.view = "setup";
      this.renderSetup();
      return;
    }
    if (this.session.status === "finished") {
      this.renderFinished();
      return;
    }
    this.renderSession();
  }

  private renderSetup(): void {
    const defaultDataset = this.datasets[0]?.path ?? "data/sample/weekly_ohlcv_synthetic.csv";
    this.root.innerHTML = `
      <div class="setup-page">
        <div class="setup-layout">
          <section class="setup-hero">
            <div class="setup-eyebrow">Weekly Decision Environment</div>
            <h1 class="setup-title">Trading<br /><span>Simulator</span></h1>
            <p class="setup-lead">
              Make investment decisions one week at a time using only information
              visible at that point in history. No look-ahead. No hindsight.
            </p>
            <div class="setup-features">
              <div class="setup-feature">
                <div class="setup-feature-icon">◈</div>
                <p class="setup-feature-title">No look-ahead</p>
                <p class="setup-feature-desc">Prices and history are capped at the current decision week.</p>
              </div>
              <div class="setup-feature">
                <div class="setup-feature-icon">◎</div>
                <p class="setup-feature-title">Weekly decisions</p>
                <p class="setup-feature-desc">Build a plan, review impact, then submit for next week's open.</p>
              </div>
              <div class="setup-feature">
                <div class="setup-feature-icon">△</div>
                <p class="setup-feature-title">Risk-aware rules</p>
                <p class="setup-feature-desc">Concentration limits, stops, and turnover constraints apply live.</p>
              </div>
            </div>
            <div class="setup-stats">
              <div>
                <div class="setup-stat-value">15</div>
                <div class="setup-stat-label">Stocks</div>
              </div>
              <div>
                <div class="setup-stat-value">Weekly</div>
                <div class="setup-stat-label">Decision cadence</div>
              </div>
              <div>
                <div class="setup-stat-value">$100k</div>
                <div class="setup-stat-label">Starting NAV</div>
              </div>
            </div>
          </section>

          <section class="setup-panel">
            <div class="setup-panel-header">
              <h2 class="setup-panel-title">Configure session</h2>
              <p class="setup-panel-subtitle">Set up your participant details and choose a dataset to begin.</p>
            </div>
            <form class="setup-form-grid" id="setup-form">
              <div class="form-field">
                <label>Run mode</label>
                <div class="mode-toggle" id="mode-toggle">
                  <label>
                    <input type="radio" name="run_mode" value="human" checked />
                    Human trader
                  </label>
                  <label>
                    <input type="radio" name="run_mode" value="ai_benchmark" />
                    AI benchmark
                  </label>
                </div>
              </div>
              <div class="setup-form-row">
                <div class="form-field">
                  <label for="participant_id">Participant code</label>
                  <input id="participant_id" name="participant_id" placeholder="participant_01" />
                </div>
                <div class="form-field">
                  <label for="episode_name">Episode name</label>
                  <input id="episode_name" name="episode_name" value="pilot_episode_01" />
                </div>
              </div>
              <div class="form-field" id="condition-field">
                <label for="condition">Session type</label>
                <select id="condition" name="condition">
                  <option value="human_only">Human only</option>
                  <option value="human_with_coach_placeholder">Human + coach placeholder</option>
                </select>
              </div>
              <div class="form-field">
                <label for="dataset_path">Dataset</label>
                <select id="dataset_path" name="dataset_path">
                  ${this.datasets
                    .map(
                      (dataset) =>
                        `<option value="${dataset.path}"${dataset.path === defaultDataset ? " selected" : ""}>${dataset.label}</option>`,
                    )
                    .join("")}
                </select>
              </div>
              <div class="form-field">
                <label for="notes">Notes <span style="color:var(--app-dim);font-weight:400">(optional)</span></label>
                <textarea id="notes" name="notes" rows="2" placeholder="Researcher notes for this session…"></textarea>
              </div>
              <button class="btn btn--primary btn--launch btn--block" type="submit" ${this.loading ? "disabled" : ""}>
                ${this.loading ? "Starting session…" : "Start session →"}
              </button>
            </form>
          </section>
        </div>
      </div>`;

    const form = this.root.querySelector<HTMLFormElement>("#setup-form");
    const conditionField = this.root.querySelector<HTMLDivElement>("#condition-field");
    form?.querySelectorAll<HTMLInputElement>('input[name="run_mode"]').forEach((input) => {
      input.addEventListener("change", () => {
        if (!conditionField) return;
        const selected = form?.querySelector<HTMLInputElement>('input[name="run_mode"]:checked');
        conditionField.style.display = selected?.value === "ai_benchmark" ? "none" : "block";
      });
    });
    form?.addEventListener("submit", async (event) => {
      event.preventDefault();
      await this.handleStart(form);
    });
  }

  private async handleStart(form: HTMLFormElement): Promise<void> {
    const data = new FormData(form);
    this.loading = true;
    this.render();
    try {
      const runMode = String(data.get("run_mode") ?? "human");
      const response = await startSession({
        participant_id: String(data.get("participant_id") ?? "").trim(),
        condition: runMode === "ai_benchmark" ? "ai_benchmark" : String(data.get("condition") ?? "human_only"),
        run_mode: runMode,
        episode_name: String(data.get("episode_name") ?? "pilot_episode_01").trim(),
        dataset_path: String(data.get("dataset_path") ?? ""),
        notes: String(data.get("notes") ?? "").trim(),
      });
      this.session = response;
      this.selectedTicker = response.observation?.available_tickers[0] ?? "";
      this.view = response.status === "finished" ? "finished" : "session";
    } catch (error) {
      alert(error instanceof Error ? error.message : "Unable to start session.");
    } finally {
      this.loading = false;
      this.render();
    }
  }

  private renderSidebar(): string {
    const session = this.session;
    if (!session) return "";
    const statusLabel = session.status.replaceAll("_", " ");
    return `
      <aside class="app-sidebar">
        <div class="sidebar-card">
          <p class="sidebar-eyebrow">Session</p>
          <div class="sidebar-status-pill">${statusLabel}</div>
          <div class="sidebar-row"><span class="sidebar-label">Participant</span><span class="sidebar-value">${session.metadata.participant_id}</span></div>
          <div class="sidebar-row"><span class="sidebar-label">Session type</span><span class="sidebar-value">${session.metadata.condition_label}</span></div>
          <div class="sidebar-row"><span class="sidebar-label">Run mode</span><span class="sidebar-value">${session.run_mode === "ai_benchmark" ? "AI Benchmark" : "Human"}</span></div>
          <div class="sidebar-row"><span class="sidebar-label">Episode</span><span class="sidebar-value">${session.metadata.episode_name}</span></div>
          <div class="sidebar-row"><span class="sidebar-label">Dataset</span><span class="sidebar-value">${session.metadata.dataset_path.split("/").pop() ?? session.metadata.dataset_path}</span></div>
        </div>
        <button class="btn btn--block" id="reset-session">Start a new session</button>
      </aside>`;
  }

  private renderSession(): void {
    const session = this.session;
    const observation = session?.observation;
    if (!session || !observation) return;

    this.root.innerHTML = `
      <div class="app-shell">
        ${this.renderSidebar()}
        <main class="app-main">
          <h1 class="app-title">Trading Session</h1>
          ${session.error ? `<div class="alert alert--error">${session.error}</div>` : ""}
          ${metricGrid([
            ["Participant", session.metadata.participant_id],
            ["Mode", session.metadata.condition_label],
            ["Episode", session.metadata.episode_name],
            ["Week", String(observation.week_index + 1)],
            ["Date", observation.date],
          ]).replace('class="metric-grid"', 'class="metric-grid session-bar"')}
          ${sectionHeader("Market", "Current week visible data")}
          <div class="panel-card" id="market-panel">
            <div class="form-field">
              <label for="chart_ticker">Choose a stock to inspect</label>
              <select id="chart_ticker">
                ${observation.available_tickers
                  .map(
                    (ticker) =>
                      `<option value="${ticker}"${ticker === this.selectedTicker ? " selected" : ""}>${ticker}</option>`,
                  )
                  .join("")}
              </select>
            </div>
            <div id="market-content"></div>
          </div>
          <hr class="divider" />
          ${sectionHeader("Portfolio", "Current positions, allocation, and performance")}
          ${metricGrid([
            ["Cash", currency(session.portfolio.cash)],
            ["Invested", currency(session.portfolio.invested)],
            ["NAV", currency(session.portfolio.total_nav)],
            ["Positions", String(session.portfolio.positions)],
          ])}
          <div class="panel-grid-2">
            <div>
              ${lineChart(session.portfolio.nav_history, { label: "Equity curve" })}
              ${drawdownChart(session.portfolio.nav_history)}
              ${insightChips(portfolioInsightChips(session.portfolio))}
            </div>
            <div>${lineChart(session.portfolio.allocation.map((row) => row.weight), { stroke: "#38bdf8", label: "Allocation weights" })}</div>
          </div>
          <div class="panel-grid-2">
            <div class="panel-card">
              <h3>Current holdings</h3>
              ${holdingsPanel(session.portfolio)}
            </div>
            <div class="panel-card">
              <h3>Risk snapshot</h3>
              ${metricGrid([
                ["HHI", session.portfolio.concentration_hhi.toFixed(4)],
                ["Turnover", pct(session.portfolio.weekly_turnover)],
                ["Volatility", session.portfolio.portfolio_volatility === null ? "N/A" : pct(session.portfolio.portfolio_volatility)],
              ])}
              <h3>Pending forced sales</h3>
              ${pendingLiquidationsPanel(observation.pending_liquidations)}
            </div>
          </div>
          <hr class="divider" />
          ${sectionHeader("Trade Planner", "Build and submit your weekly decisions")}
          <div id="planner-host"></div>
          ${session.metadata.condition === "human_with_coach_placeholder" ? noteCard("AI coach not connected yet. This space is reserved for future support.", true) : ""}
          <div id="step-feedback"></div>
        </main>
      </div>`;

    const tickerSelect = this.root.querySelector<HTMLSelectElement>("#chart_ticker");
    tickerSelect?.addEventListener("change", () => {
      this.selectedTicker = tickerSelect.value;
      this.updateMarketPanel();
    });
    this.updateMarketPanel();
    this.mountPlanner(session.planner_props);
    this.renderStepFeedback();
    this.root.querySelector("#reset-session")?.addEventListener("click", () => this.reset());
  }

  private updateMarketPanel(): void {
    const session = this.session;
    const host = this.root.querySelector<HTMLDivElement>("#market-content");
    if (!session?.observation || !host) return;
    host.innerHTML = marketPanel(session.observation, this.selectedTicker);
  }

  private mountPlanner(props: TradePlannerProps | undefined): void {
    const host = this.root.querySelector<HTMLDivElement>("#planner-host");
    if (!host) return;
    host.innerHTML = "";
    this.plannerHost = host;
    this.planner = new TradePlannerApp(host, {
      emit: async (payload: TradePlannerEventPayload) => {
        if (!this.session) return;
        try {
          this.session = await sendPlannerEvent(this.session.session, payload);
          if (this.session.status === "finished") {
            this.view = "finished";
            this.render();
            return;
          }
          this.renderSession();
        } catch (error) {
          if (this.session) {
            this.session = {
              ...this.session,
              error: error instanceof Error ? error.message : "Planner request failed.",
            };
          }
          this.renderSession();
        }
      },
      setFrameHeight: () => undefined,
    });
    if (props) this.planner.setProps(props);
  }

  private renderStepFeedback(): void {
    const host = this.root.querySelector<HTMLDivElement>("#step-feedback");
    const info = this.session?.last_step_info;
    if (!host || !info) return;
    const items = Array.isArray(info.position_change_items)
      ? (info.position_change_items as string[])
      : [];
    host.innerHTML = `
      <hr class="divider" />
      ${sectionHeader("Last week result", "What changed after your submitted plan executed")}
      <div class="panel-card">
        <p><strong>NAV:</strong> ${currency(Number(info.total_nav_before ?? 0))} → ${currency(Number(info.total_nav_after ?? 0))}</p>
        <p><strong>Cash:</strong> ${currency(Number(info.cash_before ?? 0))} → ${currency(Number(info.cash_after ?? 0))}</p>
        ${items.length ? `<ul>${items.map((item) => `<li>${item}</li>`).join("")}</ul>` : noteCard("No position changes were recorded for the last step.", true)}
      </div>`;
  }

  private renderFinished(): void {
    const session = this.session;
    if (!session) return;
    const metrics = session.metrics;
    this.root.innerHTML = `
      <div class="app-shell">
        ${this.renderSidebar()}
        <main class="app-main">
          <h1 class="app-title">${session.run_mode === "ai_benchmark" ? "AI Benchmark Simulation" : "Trading Session"}</h1>
          <p class="app-caption">Session complete. Review the summary below.</p>
          ${sectionHeader("Session complete", "Final portfolio and research metrics")}
          ${metrics ? metricGrid([
            ["Final portfolio value", currency(session.portfolio.total_nav)],
            ["Total return", pct(metrics.total_return)],
            ["Largest drawdown", pct(metrics.max_drawdown)],
            ["Realized volatility", metrics.realized_vol === null ? "N/A" : pct(metrics.realized_vol)],
            ["Average weekly turnover", pct(metrics.avg_weekly_turnover)],
            ["Average concentration", metrics.avg_hhi.toFixed(4)],
            ["Blow-up flag", metrics.blow_up_flag ? "Yes" : "No"],
          ]) : ""}
          ${dataTable(
            ["Field", "Value"],
            [
              ["Participant code", session.metadata.participant_id],
              ["Session type", session.metadata.condition_label],
              ["Episode name", session.metadata.episode_name],
              ["Dataset path", session.metadata.dataset_path],
              ["First decision week", String(session.metadata.decision_start_week)],
              ["Visible history at start", `${session.metadata.visible_history_weeks_at_start} week(s)`],
              ["Started at", session.metadata.started_at],
              ["Finished at", session.metadata.finished_at ?? "Not finished"],
            ],
          )}
          <hr class="divider" />
          ${sectionHeader("Final portfolio review", "Equity path, ending allocation, and holdings")}
          <div class="panel-grid-2">
            <div>
              ${lineChart(session.portfolio.nav_history, { label: "Equity curve" })}
              ${insightChips(portfolioInsightChips(session.portfolio))}
            </div>
            <div class="panel-card">${holdingsPanel(session.portfolio)}</div>
          </div>
        </main>
      </div>`;
    this.root.querySelector("#reset-session")?.addEventListener("click", () => this.reset());
  }

  private reset(): void {
    this.session = null;
    this.planner = null;
    this.plannerHost = null;
    this.view = "setup";
    this.render();
  }
}
