const slug = (value) => value.replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_|_$/g, "") || "session";
const exportFilename = (session, ext) => {
    const stamp = (session.metadata.finished_at ?? session.metadata.started_at)
        .replace(/[:.]/g, "-")
        .slice(0, 19);
    return `${slug(session.metadata.participant_id)}_${slug(session.metadata.episode_name)}_${stamp}.${ext}`;
};
const csvEscape = (value) => {
    if (value === null || value === undefined)
        return "";
    const text = String(value);
    if (/[",\n\r]/.test(text)) {
        return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
};
const tableToCsv = (headers, rows) => {
    const headerLine = headers.map(csvEscape).join(",");
    const body = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
    return body ? `${headerLine}\n${body}` : headerLine;
};
const appendSection = (lines, title, headers, rows) => {
    lines.push(`# ${title}`);
    lines.push(tableToCsv(headers, rows));
    lines.push("");
};
export const buildSessionExport = (session) => ({
    exported_at: new Date().toISOString(),
    export_version: 1,
    status: session.status,
    run_mode: session.run_mode,
    metadata: session.metadata,
    metrics: session.metrics ?? null,
    portfolio: session.portfolio,
    last_step_info: session.last_step_info ?? null,
    llm_decision_log: session.llm_decision_log ?? null,
});
export const buildSessionCsv = (session) => {
    const lines = [];
    const { metadata, metrics, portfolio } = session;
    appendSection(lines, "session_metadata", ["field", "value"], [
        ["participant_id", metadata.participant_id],
        ["condition", metadata.condition],
        ["condition_label", metadata.condition_label],
        ["episode_name", metadata.episode_name],
        ["dataset_path", metadata.dataset_path],
        ["run_mode", session.run_mode],
        ["started_at", metadata.started_at],
        ["finished_at", metadata.finished_at ?? ""],
        ["decision_start_week", metadata.decision_start_week],
        ["visible_history_weeks_at_start", metadata.visible_history_weeks_at_start],
        ["notes", metadata.notes ?? ""],
    ]);
    if (metrics) {
        appendSection(lines, "metrics", ["metric", "value"], [
            ["total_return", metrics.total_return],
            ["max_drawdown", metrics.max_drawdown],
            ["realized_vol", metrics.realized_vol ?? ""],
            ["sharpe_ratio", metrics.sharpe_ratio ?? ""],
            ["avg_weekly_turnover", metrics.avg_weekly_turnover],
            ["avg_hhi", metrics.avg_hhi],
            ["blow_up_flag", metrics.blow_up_flag],
            ["n_invalid_attempts", metrics.n_invalid_attempts],
            ["n_clipped_trades", metrics.n_clipped_trades],
            ["n_stop_triggers", metrics.n_stop_triggers],
            ["n_gap_adjustments", metrics.n_gap_adjustments],
            ["vol_rule_activation_week", metrics.vol_rule_activation_week ?? ""],
        ]);
    }
    appendSection(lines, "portfolio_summary", ["field", "value"], [
        ["cash", portfolio.cash],
        ["total_nav", portfolio.total_nav],
        ["invested", portfolio.invested],
        ["positions", portfolio.positions],
        ["weekly_turnover", portfolio.weekly_turnover],
        ["concentration_hhi", portfolio.concentration_hhi],
        ["portfolio_volatility", portfolio.portfolio_volatility ?? ""],
    ]);
    appendSection(lines, "equity_curve", ["decision_step", "nav"], portfolio.nav_history.map((nav, index) => [index + 1, nav]));
    appendSection(lines, "holdings", ["ticker", "shares", "average_cost", "market_value", "weight", "active_stop"], portfolio.holdings.map((row) => [
        row.ticker,
        row.shares,
        row.average_cost,
        row.market_value,
        row.weight,
        row.active_stop ?? "",
    ]));
    appendSection(lines, "allocation", ["label", "value", "weight"], portfolio.allocation.map((row) => [row.label, row.value, row.weight]));
    return `${lines.join("\n").trim()}\n`;
};
export const downloadTextFile = (filename, content, mime) => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
};
export const downloadSessionJson = (session) => {
    downloadTextFile(exportFilename(session, "json"), `${JSON.stringify(buildSessionExport(session), null, 2)}\n`, "application/json;charset=utf-8");
};
export const downloadSessionCsv = (session) => {
    downloadTextFile(exportFilename(session, "csv"), buildSessionCsv(session), "text/csv;charset=utf-8");
};
