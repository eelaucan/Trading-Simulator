import type { ObservationPayload, PortfolioPayload } from "../api/client.js";
import { currency, pct } from "../utils/format.js";

export const escapeHtml = (value: string): string =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

export const sectionHeader = (title: string, subtitle: string): string => `
  <div class="section-shell">
    <p class="section-kicker">${escapeHtml(title)}</p>
    <p class="section-subtitle">${escapeHtml(subtitle)}</p>
  </div>`;

export const zoneHeader = (title: string, subtitle: string): string => `
  <div class="zone-header">
    <h2 class="zone-title">${escapeHtml(title)}</h2>
    <p class="zone-subtitle">${escapeHtml(subtitle)}</p>
  </div>`;

export const allocationBars = (
  rows: Array<{ label: string; weight: number }>,
): string => {
  if (!rows.length) return noteCard("No allocation data yet.", true);
  const bars = rows
    .map(
      (row) => `
      <div class="alloc-row">
        <div class="alloc-meta">
          <span class="alloc-label">${escapeHtml(row.label)}</span>
          <span class="alloc-pct">${escapeHtml(pct(row.weight))}</span>
        </div>
        <div class="alloc-track"><div class="alloc-fill" style="width:${Math.max(2, row.weight * 100)}%"></div></div>
      </div>`,
    )
    .join("");
  return `<div class="alloc-list">${bars}</div>`;
};

export const metricCard = (label: string, value: string): string => `
  <div class="metric-card">
    <div class="metric-label">${escapeHtml(label)}</div>
    <div class="metric-value">${escapeHtml(value)}</div>
  </div>`;

export const metricGrid = (items: Array<[string, string]>): string =>
  `<div class="metric-grid">${items.map(([label, value]) => metricCard(label, value)).join("")}</div>`;

export const noteCard = (text: string, quiet = false): string =>
  `<div class="note-card${quiet ? " note-card--quiet" : ""}">${escapeHtml(text)}</div>`;

export const insightChips = (items: Array<[string, string]>): string =>
  items
    .map(
      ([label, tone]) =>
        `<span class="insight-chip insight-chip--${escapeHtml(tone)}">${escapeHtml(label)}</span>`,
    )
    .join("");

export const dataTable = (headers: string[], rows: string[][]): string => {
  if (!rows.length) return noteCard("No rows to display.", true);
  return `
    <table class="data-table">
      <thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}
      </tbody>
    </table>`;
};

export const lineChart = (
  values: number[],
  options?: { stroke?: string; label?: string },
): string => {
  if (values.length < 2) return noteCard("Not enough data for a chart yet.", true);
  const width = 640;
  const height = 180;
  const padding = 18;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-9);
  const points = values
    .map((value, index) => {
      const x = padding + (index / (values.length - 1)) * (width - padding * 2);
      const y = height - padding - ((value - min) / span) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");
  return `
    <div class="chart-wrap">
      <div class="chart-title">${escapeHtml(options?.label ?? "Chart")}</div>
      <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
        <polyline fill="none" stroke="${options?.stroke ?? "#22c55e"}" stroke-width="2.5" points="${points}" />
      </svg>
    </div>`;
};

export const drawdownChart = (values: number[]): string => {
  if (values.length < 2) return noteCard("Drawdown becomes visible after more weeks.", true);
  let peak = values[0];
  const drawdowns = values.map((value) => {
    peak = Math.max(peak, value);
    return peak > 0 ? value / peak - 1 : 0;
  });
  return lineChart(drawdowns, { stroke: "#ef4444", label: "Drawdown so far" });
};

export const portfolioInsightChips = (portfolio: PortfolioPayload): Array<[string, string]> => {
  const chips: Array<[string, string]> = [];
  const cashWeight = portfolio.total_nav > 0 ? portfolio.cash / portfolio.total_nav : 0;
  const maxWeight = portfolio.holdings.reduce(
    (max, row) => Math.max(max, row.weight),
    0,
  );
  if (portfolio.positions === 0) {
    return [["No holdings yet", "neutral"], ["All capital is currently in cash", "good"]];
  }
  if (cashWeight > 0.65) chips.push(["Most capital is still in cash", "neutral"]);
  else if (cashWeight < 0.2) chips.push(["Most capital is currently invested", "good"]);
  if (maxWeight > 0.5) chips.push(["Portfolio is concentrated in one position", "risk"]);
  else if (portfolio.positions >= 3 && maxWeight < 0.35) {
    chips.push(["Portfolio is currently diversified", "good"]);
  }
  if (portfolio.nav_history.length >= 2) {
    const peak = Math.max(...portfolio.nav_history);
    if (portfolio.total_nav >= peak * 0.99) chips.push(["Portfolio value is near its visible high", "good"]);
    else chips.push(["Portfolio remains below its visible peak", "warn"]);
  }
  return chips;
};

export const marketPanel = (observation: ObservationPayload, selectedTicker: string): string => {
  const history = observation.price_history.filter((row) => row.ticker === selectedTicker);
  const selected = observation.market_rows.find((row) => row.ticker === selectedTicker);
  const previousClose =
    history.length >= 2 ? history[history.length - 2].close : null;
  const change =
    selected && previousClose ? selected.close - previousClose : null;
  const historyValues = history.map((row) => row.close);
  const marketRows = observation.market_rows.map((row) => [
    row.ticker,
    currency(row.close),
    row.change_vs_previous_close === null ? "N/A" : pct(row.change_vs_previous_close),
    currency(row.low),
    currency(row.high),
    `${row.volume.toLocaleString()}`,
  ]);
  const weeklyChange =
    selected?.change_vs_previous_close !== null && selected?.change_vs_previous_close !== undefined
      ? pct(selected.change_vs_previous_close)
      : "N/A";

  return `
    <div class="market-detail-grid">
      <div class="market-detail-main">
        ${metricGrid([
          ["Close", selected ? currency(selected.close) : "N/A"],
          ["Open", selected ? currency(selected.open) : "N/A"],
          ["Week change", weeklyChange],
          ["Visible weeks", `${history.length}`],
        ])}
        ${lineChart(historyValues, { label: `${selectedTicker} price history (visible only)` })}
      </div>
      <div class="market-detail-aside">
        ${selected ? noteCard(`${selectedTicker} range this week: ${currency(selected.low)} – ${currency(selected.high)}`, true) : ""}
        ${change !== null ? noteCard(`Change vs prior visible close: ${currency(change)}`, true) : ""}
      </div>
    </div>
    <div class="table-wrap">
      ${dataTable(["Stock", "Close", "Change", "Low", "High", "Volume"], marketRows)}
    </div>
  `;
};

export const holdingsPanel = (portfolio: PortfolioPayload): string => {
  if (!portfolio.holdings.length) {
    return noteCard("You currently hold only cash. No stock positions are open right now.", true);
  }
  const rows = portfolio.holdings.map((row) => [
    row.ticker,
    row.shares.toFixed(4),
    currency(row.average_cost),
    currency(row.market_value),
    pct(row.weight),
    row.active_stop === null ? "None" : currency(row.active_stop),
  ]);
  return dataTable(
    ["Stock", "Shares Held", "Average Cost", "Current Market Value", "Portfolio Weight", "Active Stop"],
    rows,
  );
};

export const pendingLiquidationsPanel = (
  items: ObservationPayload["pending_liquidations"],
): string => {
  if (!items.length) {
    return noteCard("No forced sale is currently scheduled for a future week.", true);
  }
  const rows = items.map((item) => [
    item.ticker,
    currency(item.triggered_by_low),
    currency(item.stop_level),
    String(item.execution_week),
  ]);
  return dataTable(
    ["Stock", "Low That Triggered It", "Stop Price", "Scheduled Execution Week"],
    rows,
  );
};
