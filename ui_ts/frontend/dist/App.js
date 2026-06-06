import { advanceGeminiStep, fetchDatasets, sendPlannerEvent, startSession, } from "./api/client.js";
import { allocationBars, dataTable, drawdownChart, holdingsPanel, insightChips, lineChart, marketPanel, metricGrid, noteCard, pendingLiquidationsPanel, portfolioInsightChips, zoneHeader, } from "./components/ui.js";
import { TradePlannerApp } from "./TradePlanner.js";
import { downloadSessionCsv, downloadSessionJson } from "./utils/export.js";
import { currency, pct } from "./utils/format.js";
export class TradingSimulatorApp {
    constructor(root) {
        this.view = "setup";
        this.datasets = [];
        this.session = null;
        this.selectedTicker = "";
        this.planner = null;
        this.plannerHost = null;
        this.loading = false;
        this.geminiStatus = "";
        this.root = root;
    }
    async init() {
        try {
            this.datasets = await fetchDatasets();
        }
        catch {
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
    render() {
        if (this.view === "setup") {
            this.renderSetup();
            return;
        }
        if (!this.session) {
            this.view = "setup";
            this.renderSetup();
            return;
        }
        if (this.view === "gemini_running") {
            this.renderGeminiRunning();
            return;
        }
        if (this.session.status === "finished") {
            this.renderFinished();
            return;
        }
        this.renderSession();
    }
    renderSetup() {
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
                    Rule-based AI
                  </label>
                  <label>
                    <input type="radio" name="run_mode" value="ai_gemini" />
                    Gemini AI
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
            .map((dataset) => `<option value="${dataset.path}"${dataset.path === defaultDataset ? " selected" : ""}>${dataset.label}</option>`)
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
        const form = this.root.querySelector("#setup-form");
        const conditionField = this.root.querySelector("#condition-field");
        form?.querySelectorAll('input[name="run_mode"]').forEach((input) => {
            input.addEventListener("change", () => {
                if (!conditionField)
                    return;
                const selected = form?.querySelector('input[name="run_mode"]:checked');
                const isAutonomousAi = selected?.value === "ai_benchmark" || selected?.value === "ai_gemini";
                conditionField.style.display = isAutonomousAi ? "none" : "block";
            });
        });
        form?.addEventListener("submit", async (event) => {
            event.preventDefault();
            await this.handleStart(form);
        });
    }
    async handleStart(form) {
        const data = new FormData(form);
        this.loading = true;
        this.render();
        try {
            const runMode = String(data.get("run_mode") ?? "human");
            const response = await startSession({
                participant_id: String(data.get("participant_id") ?? "").trim(),
                condition: runMode === "ai_gemini"
                    ? "ai_gemini"
                    : runMode === "ai_benchmark"
                        ? "ai_benchmark"
                        : String(data.get("condition") ?? "human_only"),
                run_mode: runMode,
                episode_name: String(data.get("episode_name") ?? "pilot_episode_01").trim(),
                dataset_path: String(data.get("dataset_path") ?? ""),
                notes: String(data.get("notes") ?? "").trim(),
            });
            this.session = response;
            this.selectedTicker = response.observation?.available_tickers[0] ?? "";
            if (runMode === "ai_gemini" && response.status === "running") {
                this.view = "gemini_running";
                this.render();
                await this.runGeminiEpisode();
                return;
            }
            this.view = response.status === "finished" ? "finished" : "session";
        }
        catch (error) {
            alert(error instanceof Error ? error.message : "Unable to start session.");
        }
        finally {
            this.loading = false;
            this.render();
        }
    }
    async runGeminiEpisode() {
        while (this.session?.status === "running") {
            const week = (this.session.observation?.week_index ?? 0) + 1;
            this.geminiStatus = `Gemini is deciding for week ${week}…`;
            this.render();
            try {
                this.session = await advanceGeminiStep(this.session.session);
            }
            catch (error) {
                alert(error instanceof Error ? error.message : "Gemini step failed.");
                this.view = "setup";
                this.session = null;
                this.geminiStatus = "";
                return;
            }
        }
        this.geminiStatus = "";
        this.view = this.session?.status === "finished" ? "finished" : "setup";
    }
    renderGeminiRunning() {
        const session = this.session;
        this.root.innerHTML = `
      <div class="setup-page">
        <div class="setup-layout setup-layout--centered">
          <section class="setup-panel">
            <div class="setup-panel-header">
              <h2 class="setup-panel-title">Gemini AI session running</h2>
              <p class="setup-panel-subtitle">
                The model makes one weekly decision at a time using the same market
                information and risk rules as human participants.
              </p>
            </div>
            <div class="panel-card">
              <p class="panel-label">Progress</p>
              <p class="gemini-status">${this.geminiStatus || "Starting…"}</p>
              ${session?.metadata ? `<p class="gemini-meta">Participant: ${session.metadata.participant_id} · Episode: ${session.metadata.episode_name}</p>` : ""}
            </div>
          </section>
        </div>
      </div>`;
    }
    renderSidebar() {
        const session = this.session;
        if (!session)
            return "";
        const weekLabel = session.observation !== undefined
            ? `Week ${session.observation.week_index + 1}`
            : session.status === "finished"
                ? "Complete"
                : "—";
        return `
      <aside class="app-sidebar">
        <div class="sidebar-brand">Trading Simulator</div>
        <div class="sidebar-card">
          <p class="sidebar-eyebrow">Current step</p>
          <div class="sidebar-week">${weekLabel}</div>
          <div class="sidebar-row"><span class="sidebar-label">Participant</span><span class="sidebar-value">${session.metadata.participant_id}</span></div>
          <div class="sidebar-row"><span class="sidebar-label">Type</span><span class="sidebar-value">${session.metadata.condition_label}</span></div>
          <div class="sidebar-row"><span class="sidebar-label">Episode</span><span class="sidebar-value">${session.metadata.episode_name}</span></div>
          <div class="sidebar-row"><span class="sidebar-label">Dataset</span><span class="sidebar-value">${session.metadata.dataset_path.split("/").pop() ?? session.metadata.dataset_path}</span></div>
        </div>
        <button class="btn btn--block" id="reset-session">New session</button>
      </aside>`;
    }
    renderSessionHeader(observation, portfolio) {
        return `
      <header class="session-header">
        <div class="session-header-primary">
          <span class="session-header-label">Decision week</span>
          <span class="session-header-value">${observation.week_index + 1}</span>
        </div>
        <div class="session-header-stat">
          <span class="session-header-label">As of</span>
          <span class="session-header-value">${observation.date}</span>
        </div>
        <div class="session-header-stat">
          <span class="session-header-label">Portfolio NAV</span>
          <span class="session-header-value session-header-value--nav">${currency(portfolio.total_nav)}</span>
        </div>
        <div class="session-header-stat">
          <span class="session-header-label">Cash</span>
          <span class="session-header-value">${currency(portfolio.cash)}</span>
        </div>
      </header>`;
    }
    renderSession() {
        const session = this.session;
        const observation = session?.observation;
        if (!session || !observation)
            return;
        this.root.innerHTML = `
      <div class="app-shell">
        ${this.renderSidebar()}
        <main class="app-main session-main">
          ${session.error ? `<div class="alert alert--error">${session.error}</div>` : ""}
          ${this.renderSessionHeader(observation, session.portfolio)}

          <section class="session-zone">
            <div class="zone-header zone-header--row">
              <div>
                <h2 class="zone-title">Market</h2>
                <p class="zone-subtitle">Prices visible through the end of this week only</p>
              </div>
              <div class="form-field form-field--inline">
                <label for="chart_ticker">Stock</label>
                <select id="chart_ticker">
                  ${observation.available_tickers
            .map((ticker) => `<option value="${ticker}"${ticker === this.selectedTicker ? " selected" : ""}>${ticker}</option>`)
            .join("")}
                </select>
              </div>
            </div>
            <div class="panel-card" id="market-panel">
              <div id="market-content"></div>
            </div>
          </section>

          <section class="session-zone">
            ${zoneHeader("Portfolio", "Positions, performance, and risk")}
            <div class="session-portfolio-top">
              ${metricGrid([
            ["Invested", currency(session.portfolio.invested)],
            ["Positions", String(session.portfolio.positions)],
            ["HHI", session.portfolio.concentration_hhi.toFixed(4)],
            ["Weekly turnover", pct(session.portfolio.weekly_turnover)],
        ])}
              <div class="panel-card panel-card--compact">
                <p class="panel-label">Allocation</p>
                ${allocationBars(session.portfolio.allocation)}
              </div>
            </div>
            <div class="session-charts-row">
              ${lineChart(session.portfolio.nav_history, { label: "Equity curve" })}
              ${drawdownChart(session.portfolio.nav_history)}
            </div>
            ${insightChips(portfolioInsightChips(session.portfolio))}
            <div class="session-tables-row">
              <div class="panel-card">
                <p class="panel-label">Holdings</p>
                ${holdingsPanel(session.portfolio)}
              </div>
              <div class="panel-card">
                <p class="panel-label">Risk &amp; scheduled liquidations</p>
                ${metricGrid([
            ["Volatility", session.portfolio.portfolio_volatility === null ? "N/A" : pct(session.portfolio.portfolio_volatility)],
            ["Concentration", session.portfolio.concentration_hhi.toFixed(4)],
            ["Turnover", pct(session.portfolio.weekly_turnover)],
        ])}
                ${pendingLiquidationsPanel(observation.pending_liquidations)}
              </div>
            </div>
          </section>

          <section class="session-zone session-zone--planner">
            ${zoneHeader("Trade planner", "Build your plan for next week's open, then submit")}
            <div id="planner-host" class="planner-host"></div>
            ${session.metadata.condition === "human_with_coach_placeholder" ? noteCard("Coach panel reserved for a future release.", true) : ""}
          </section>

          <div id="step-feedback"></div>
        </main>
      </div>`;
        const tickerSelect = this.root.querySelector("#chart_ticker");
        tickerSelect?.addEventListener("change", () => {
            this.selectedTicker = tickerSelect.value;
            this.updateMarketPanel();
        });
        this.updateMarketPanel();
        this.mountPlanner(session.planner_props);
        this.renderStepFeedback();
        this.root.querySelector("#reset-session")?.addEventListener("click", () => this.reset());
    }
    updateMarketPanel() {
        const session = this.session;
        const host = this.root.querySelector("#market-content");
        if (!session?.observation || !host)
            return;
        host.innerHTML = marketPanel(session.observation, this.selectedTicker);
    }
    mountPlanner(props) {
        const host = this.root.querySelector("#planner-host");
        if (!host)
            return;
        host.innerHTML = "";
        this.plannerHost = host;
        this.planner = new TradePlannerApp(host, {
            emit: async (payload) => {
                if (!this.session)
                    return;
                try {
                    this.session = await sendPlannerEvent(this.session.session, payload);
                    if (this.session.status === "finished") {
                        this.view = "finished";
                        this.render();
                        return;
                    }
                    this.renderSession();
                }
                catch (error) {
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
        if (props)
            this.planner.setProps(props);
    }
    renderStepFeedback() {
        const host = this.root.querySelector("#step-feedback");
        const info = this.session?.last_step_info;
        if (!host || !info)
            return;
        const items = Array.isArray(info.position_change_items)
            ? info.position_change_items
            : [];
        host.innerHTML = `
      <section class="session-zone">
        ${zoneHeader("Last week", "Results after your submitted plan executed")}
        <div class="panel-card feedback-panel">
          <div class="feedback-metrics">
            <div><span class="feedback-label">NAV</span><span class="feedback-value">${currency(Number(info.total_nav_before ?? 0))} → ${currency(Number(info.total_nav_after ?? 0))}</span></div>
            <div><span class="feedback-label">Cash</span><span class="feedback-value">${currency(Number(info.cash_before ?? 0))} → ${currency(Number(info.cash_after ?? 0))}</span></div>
          </div>
          ${items.length ? `<ul class="feedback-list">${items.map((item) => `<li>${item}</li>`).join("")}</ul>` : noteCard("No position changes recorded.", true)}
        </div>
      </section>`;
    }
    renderGeminiSummary(session) {
        const summary = session.gemini_summary;
        if (session.run_mode !== "ai_gemini" || !summary)
            return "";
        const allFallback = summary.fallback_weeks === summary.decisions && summary.decisions > 0;
        return `
      <section class="session-zone">
        ${zoneHeader("Gemini decisions", "Weekly model calls and fallback status")}
        <div class="panel-card ${allFallback ? "panel-card--warning" : ""}">
          ${allFallback ? `<div class="alert alert--error">Gemini did not execute trades this session. Every week fell back to HOLD.${summary.last_error ? ` Last error: ${summary.last_error}` : ""}</div>` : ""}
          ${metricGrid([
            ["Decision weeks", String(summary.decisions)],
            ["Weeks with trades", String(summary.trade_weeks)],
            ["Fallback HOLD weeks", String(summary.fallback_weeks)],
            ["Signal rescue weeks", String(summary.signal_rescue_weeks ?? 0)],
            ["Last error", summary.last_error ?? "None"],
        ])}
        </div>
      </section>`;
    }
    renderFinished() {
        const session = this.session;
        if (!session)
            return;
        const metrics = session.metrics;
        this.root.innerHTML = `
      <div class="app-shell">
        ${this.renderSidebar()}
        <main class="app-main session-main">
          <header class="session-header">
            <div class="session-header-primary">
              <span class="session-header-label">Status</span>
              <span class="session-header-value">Complete</span>
            </div>
            <div class="session-header-stat">
              <span class="session-header-label">Final NAV</span>
              <span class="session-header-value session-header-value--nav">${currency(session.portfolio.total_nav)}</span>
            </div>
            ${metrics ? `
            <div class="session-header-stat">
              <span class="session-header-label">Total return</span>
              <span class="session-header-value">${pct(metrics.total_return)}</span>
            </div>
            <div class="session-header-stat">
              <span class="session-header-label">Max drawdown</span>
              <span class="session-header-value">${pct(metrics.max_drawdown)}</span>
            </div>` : ""}
          </header>

          <section class="session-zone">
            <div class="zone-header zone-header--row">
              <div>
                <h2 class="zone-title">Export results</h2>
                <p class="zone-subtitle">Save a copy to your computer before starting a new session.</p>
              </div>
              <div class="export-actions">
                <button type="button" class="btn" id="download-json">Download JSON</button>
                <button type="button" class="btn" id="download-csv">Download CSV</button>
              </div>
            </div>
          </section>

          ${this.renderGeminiSummary(session)}

          <section class="session-zone">
            ${zoneHeader("Results", "Research metrics for this session")}
            ${metrics ? metricGrid([
            ["Final portfolio value", currency(session.portfolio.total_nav)],
            ["Total return", pct(metrics.total_return)],
            ["Largest drawdown", pct(metrics.max_drawdown)],
            ["Realized volatility", metrics.realized_vol === null ? "N/A" : pct(metrics.realized_vol)],
            ["Average weekly turnover", pct(metrics.avg_weekly_turnover)],
            ["Average concentration", metrics.avg_hhi.toFixed(4)],
            ["Blow-up flag", metrics.blow_up_flag ? "Yes" : "No"],
        ]) : ""}
            <div class="panel-card panel-card--spaced">
              <p class="panel-label">Session details</p>
              ${dataTable(["Field", "Value"], [
            ["Participant code", session.metadata.participant_id],
            ["Session type", session.metadata.condition_label],
            ["Episode name", session.metadata.episode_name],
            ["Dataset path", session.metadata.dataset_path],
            ["First decision week", String(session.metadata.decision_start_week)],
            ["Visible history at start", `${session.metadata.visible_history_weeks_at_start} week(s)`],
            ["Started at", session.metadata.started_at],
            ["Finished at", session.metadata.finished_at ?? "Not finished"],
        ])}
            </div>
          </section>

          <section class="session-zone">
            ${zoneHeader("Final portfolio", "Equity path and ending positions")}
            <div class="session-charts-row">
              ${lineChart(session.portfolio.nav_history, { label: "Equity curve" })}
              <div class="panel-card panel-card--compact">
                <p class="panel-label">Allocation</p>
                ${allocationBars(session.portfolio.allocation)}
              </div>
            </div>
            ${insightChips(portfolioInsightChips(session.portfolio))}
            <div class="panel-card panel-card--spaced">
              <p class="panel-label">Holdings</p>
              ${holdingsPanel(session.portfolio)}
            </div>
          </section>
        </main>
      </div>`;
        this.root.querySelector("#reset-session")?.addEventListener("click", () => this.reset());
        this.root.querySelector("#download-json")?.addEventListener("click", () => {
            if (this.session)
                downloadSessionJson(this.session);
        });
        this.root.querySelector("#download-csv")?.addEventListener("click", () => {
            if (this.session)
                downloadSessionCsv(this.session);
        });
    }
    reset() {
        this.session = null;
        this.planner = null;
        this.plannerHost = null;
        this.view = "setup";
        this.render();
    }
}
