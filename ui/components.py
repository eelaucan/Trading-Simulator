"""Participant-facing Streamlit components for the weekly trading task."""

from __future__ import annotations

from dataclasses import dataclass
import html
from typing import Any, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from simulator.actions import Action, ActionType, QuantityType, ValidationResult
from simulator.config import SimulatorConfig
from simulator.metrics import SimulationMetrics
from simulator.observation import Observation, PendingLiquidation
from simulator.state import PortfolioState
from ui.session import SessionMetadata, SUPPORTED_CONDITIONS, condition_display_label


_ACTION_LABELS: dict[ActionType, str] = {
    ActionType.BUY: "Buy shares",
    ActionType.SELL: "Sell shares",
    ActionType.REDUCE: "Reduce a holding",
    ActionType.SET_STOP: "Set a stop price",
    ActionType.REMOVE_STOP: "Remove a stop price",
    ActionType.HOLD: "Do nothing this week",
}

_ACTION_HELP_TEXT: dict[ActionType, str] = {
    ActionType.BUY: "Choose a stock and decide how large your purchase should be.",
    ActionType.SELL: "Choose a stock and decide how much of it to sell.",
    ActionType.REDUCE: "Trim an existing holding by a percentage.",
    ActionType.SET_STOP: "Add a stop price that can schedule a forced sale if the market drops below it.",
    ActionType.REMOVE_STOP: "Remove an existing stop price from one of your holdings.",
    ActionType.HOLD: "Make no new trading decision this week.",
}

_PREVIEW_ACTION_ORDER: dict[ActionType, int] = {
    ActionType.SELL: 0,
    ActionType.REDUCE: 0,
    ActionType.BUY: 1,
    ActionType.SET_STOP: 2,
    ActionType.REMOVE_STOP: 2,
    ActionType.HOLD: 3,
}
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class PlanImpactPreview:
    """Lightweight participant-facing estimate for the current weekly plan."""

    estimated_spend: float
    estimated_proceeds: float
    estimated_transaction_costs: float
    estimated_remaining_cash: float
    estimated_positions_after: int
    estimated_invested_after: float
    projected_max_weight: float | None
    warnings: tuple[str, ...]
    notes: tuple[str, ...]


def apply_ui_theme() -> None:
    """Apply a dark chart-first terminal theme while keeping Streamlit controls usable."""
    st.markdown(
        """
        <style>
        html, body, [class*="st-"], [data-testid="stMarkdownContainer"] *, label, input, textarea, select {
            font-family: Inter, "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
        }
        h1, h2, h3, h4, .section-kicker, .trade-platform-title, .terminal-bar-value, .watchlist-title {
            font-family: "Times New Roman", Times, serif !important;
        }
        span.material-symbols-rounded,
        span.material-symbols-outlined,
        span.material-icons,
        i.material-icons,
        [data-testid="stExpanderToggleIcon"] span,
        [data-testid="stSidebarCollapsedControl"] span {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
            font-weight: normal !important;
            font-style: normal !important;
            -webkit-font-feature-settings: "liga";
            font-feature-settings: "liga";
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(47, 84, 145, 0.18), transparent 24%),
                linear-gradient(180deg, #0b1017 0%, #101720 100%);
            color: #dce6ef;
        }
        .main .block-container {
            max-width: 1540px;
            padding-top: 0.95rem;
            padding-bottom: 2.4rem;
        }
        h1, h2, h3, h4 {
            color: #f4f8fb;
            letter-spacing: 0.01em;
            margin-bottom: 0.25rem;
        }
        h1 {
            font-size: 2.28rem;
            font-weight: 700;
        }
        h2 {
            font-size: 1.46rem;
            font-weight: 700;
        }
        h3 {
            font-size: 1.03rem;
            font-weight: 700;
        }
        p, li, div, span {
            color: #b8c4d0;
            line-height: 1.45;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e1520 0%, #131b26 100%);
            border-right: 1px solid #233244;
            min-width: 276px !important;
            max-width: 276px !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-left: 0.78rem;
            padding-right: 0.78rem;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(17, 25, 35, 0.98) 0%, rgba(22, 31, 43, 0.98) 100%);
            border: 1px solid #243548;
            border-radius: 16px;
            padding: 0.84rem 0.92rem;
            box-shadow: 0 16px 28px rgba(0, 0, 0, 0.18);
            min-height: 102px;
        }
        [data-testid="stMetricLabel"] {
            color: #88a0b6;
            font-size: 0.78rem;
            line-height: 1.22;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
        }
        [data-testid="stMetricValue"] {
            color: #f4f7fb;
            font-size: 1.56rem;
            line-height: 1.1;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.88rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #243548;
            border-radius: 16px;
            overflow: hidden;
            background: rgba(17, 25, 35, 0.92);
        }
        div[data-testid="stDataFrame"] * {
            color: #dbe7f1 !important;
            background-color: transparent !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="base-input"] {
            border-radius: 14px !important;
            border: 1.5px solid #37506a !important;
            background: rgba(15, 22, 31, 0.98) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
        }
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="textarea"] > div:focus-within,
        div[data-baseweb="base-input"]:focus-within {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.16) !important;
        }
        label p {
            color: #dce6ef !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }
        input, textarea, select {
            color: #f2f6fa !important;
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid #31465a;
            background: linear-gradient(180deg, #18212b 0%, #121a24 100%);
            color: #dde7f1;
            padding: 0.64rem 1rem;
            min-height: 2.8rem;
            font-weight: 700;
            font-size: 0.93rem;
            line-height: 1.2;
            white-space: normal;
            box-shadow: 0 10px 18px rgba(0, 0, 0, 0.18);
            transition: all 0.16s ease;
        }
        .stButton > button:hover {
            border-color: #496885;
            color: #ffffff;
            transform: translateY(-1px);
            box-shadow: 0 12px 22px rgba(0, 0, 0, 0.24);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #1f8f5f 0%, #166946 100%);
            color: white;
            border-color: #166946;
            box-shadow: 0 16px 28px rgba(21, 98, 66, 0.3);
        }
        .stButton > button[kind="primary"]:hover {
            color: white;
            border-color: #1f8f5f;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 18px !important;
            border: 1px solid #243548 !important;
            background: linear-gradient(180deg, rgba(15, 22, 31, 0.96) 0%, rgba(19, 28, 38, 0.96) 100%);
            box-shadow: 0 18px 32px rgba(0, 0, 0, 0.18);
        }
        .section-shell {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
            margin: 0.12rem 0 0.75rem 0;
            box-shadow: none;
        }
        .section-kicker {
            margin: 0;
            font-size: 1.14rem;
            font-weight: 700;
            color: #f1f6fb;
        }
        .section-subtitle {
            margin: 0.22rem 0 0.15rem 0;
            color: #8da2b7;
            font-size: 0.9rem;
        }
        .note-card {
            border-radius: 14px;
            border: 1px solid #2a3d51;
            background: rgba(16, 23, 33, 0.92);
            padding: 0.75rem 0.9rem;
            margin: 0.25rem 0 0.85rem 0;
            color: #b7c6d6;
        }
        .note-card--quiet {
            background: rgba(13, 19, 28, 0.92);
        }
        .sidebar-card {
            border-radius: 16px;
            border: 1px solid #243548;
            background: rgba(17, 25, 35, 0.92);
            padding: 0.82rem 0.9rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
        }
        .sidebar-eyebrow {
            margin: 0 0 0.3rem 0;
            color: #8196ac;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
        }
        .sidebar-status-pill {
            display: inline-block;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.14);
            border: 1px solid rgba(59, 130, 246, 0.25);
            color: #d5e7ff;
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        .sidebar-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.6rem;
            margin: 0.34rem 0;
        }
        .sidebar-label {
            color: #8aa0b5;
            font-size: 0.85rem;
        }
        .sidebar-value {
            color: #edf4fb;
            font-size: 0.92rem;
            font-weight: 700;
            text-align: right;
            word-break: break-word;
        }
        .platform-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            margin: 0.1rem 0 0.8rem 0;
        }
        .trade-platform-kicker {
            color: #7e93aa;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            margin: 0;
        }
        .trade-platform-title {
            color: #f4f8fb;
            font-size: 2rem;
            line-height: 1.05;
            margin: 0.12rem 0 0.08rem 0;
        }
        .trade-platform-copy {
            color: #8ea3b8;
            margin: 0;
            font-size: 0.95rem;
        }
        .terminal-bar {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.25rem 0 1rem 0;
        }
        .terminal-bar-card {
            border-radius: 16px;
            border: 1px solid #243548;
            background: linear-gradient(180deg, rgba(16, 24, 34, 0.98) 0%, rgba(20, 29, 40, 0.98) 100%);
            padding: 0.78rem 0.84rem;
            box-shadow: 0 14px 24px rgba(0, 0, 0, 0.16);
        }
        .terminal-bar-label {
            color: #7f95ab;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.72rem;
            margin-bottom: 0.3rem;
        }
        .terminal-bar-value {
            color: #f4f8fb;
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .terminal-bar-meta {
            color: #8ea4b9;
            font-size: 0.83rem;
            margin-top: 0.42rem;
        }
        .watchlist-shell {
            border-radius: 18px;
            border: 1px solid #243548;
            background: linear-gradient(180deg, rgba(14, 20, 29, 0.98) 0%, rgba(18, 26, 36, 0.98) 100%);
            padding: 0.92rem;
            box-shadow: 0 18px 30px rgba(0, 0, 0, 0.16);
        }
        .watchlist-title {
            color: #eff5fb;
            font-size: 1.05rem;
            margin: 0;
        }
        .watchlist-copy {
            color: #8ea4ba;
            font-size: 0.88rem;
            margin: 0.24rem 0 0.8rem 0;
        }
        .watch-price {
            color: #eef5fb;
            font-weight: 700;
            font-size: 0.9rem;
            margin-top: 0.42rem;
        }
        .watch-subcopy {
            color: #7d93a8;
            font-size: 0.82rem;
            margin-top: 0.55rem;
        }
        .panel-shell {
            border-radius: 20px;
            border: 1px solid #243548;
            background: linear-gradient(180deg, rgba(14, 20, 29, 0.98) 0%, rgba(19, 28, 38, 0.98) 100%);
            padding: 1rem 1rem 0.82rem 1rem;
            box-shadow: 0 18px 30px rgba(0, 0, 0, 0.16);
        }
        .panel-title {
            color: #eef5fb;
            font-size: 1.1rem;
            font-weight: 700;
            margin: 0;
        }
        .panel-copy {
            color: #8fa5ba;
            font-size: 0.9rem;
            margin: 0.22rem 0 0.85rem 0;
        }
        .snapshot-bar {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 0.65rem;
        }
        .snapshot-symbol {
            color: #f6fbff;
            font-size: 1.42rem;
            font-weight: 700;
            margin: 0;
        }
        .snapshot-meta {
            color: #8ba1b6;
            font-size: 0.88rem;
            margin: 0.18rem 0 0 0;
        }
        .empty-plan-card {
            border-radius: 16px;
            border: 1px dashed #2d4155;
            background: rgba(11, 17, 26, 0.92);
            padding: 0.85rem 0.95rem;
            margin: 0.3rem 0 0.85rem 0;
        }
        .empty-plan-title {
            font-weight: 700;
            color: #e8f0f7;
            margin-bottom: 0.2rem;
        }
        .empty-plan-copy {
            color: #8fa4ba;
            margin: 0;
        }
        .insight-chip {
            display: inline-block;
            margin: 0 0.45rem 0.45rem 0;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            border: 1px solid #2f4256;
            background: rgba(19, 28, 39, 0.92);
            color: #c3d1de;
            font-size: 0.92rem;
        }
        .insight-chip--good {
            background: rgba(31, 143, 95, 0.14);
            border-color: rgba(31, 143, 95, 0.28);
            color: #64e6a5;
        }
        .insight-chip--warn {
            background: rgba(245, 158, 11, 0.14);
            border-color: rgba(245, 158, 11, 0.28);
            color: #f7c86c;
        }
        .insight-chip--risk {
            background: rgba(239, 68, 68, 0.12);
            border-color: rgba(239, 68, 68, 0.24);
            color: #ff8c8c;
        }
        .insight-chip--neutral {
            background: rgba(55, 78, 103, 0.18);
            border-color: rgba(87, 114, 141, 0.28);
            color: #bfd2e2;
        }
        .feedback-card {
            border-radius: 16px;
            padding: 0.82rem 0.95rem;
            margin: 0.15rem 0 0.65rem 0;
            background: rgba(17, 25, 35, 0.96);
            border: 1px solid #243548;
            box-shadow: 0 14px 24px rgba(0, 0, 0, 0.16);
        }
        .feedback-card--success { border-left: 5px solid #1f8f5f; }
        .feedback-card--warning { border-left: 5px solid #d39a27; }
        .feedback-card--error { border-left: 5px solid #dc5d5d; }
        .feedback-card--neutral { border-left: 5px solid #607a96; }
        .feedback-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #f0f6fc;
            margin-bottom: 0.35rem;
        }
        .feedback-card ul {
            margin: 0.2rem 0 0 1rem;
            padding: 0;
        }
        .feedback-card li {
            margin: 0.2rem 0;
            color: #bfd0de;
        }
        .micro-copy {
            color: #8ea1b4;
            font-size: 0.88rem;
            margin-top: 0.2rem;
        }
        button[role="tab"] {
            background: rgba(15, 22, 31, 0.92) !important;
            border: 1px solid #27384a !important;
            border-radius: 12px !important;
            color: #99aec3 !important;
        }
        button[role="tab"][aria-selected="true"] {
            background: rgba(25, 36, 49, 0.98) !important;
            border-color: #38506a !important;
            color: #f1f6fb !important;
        }
        @media (max-width: 1200px) {
            .terminal-bar {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
        }
        @media (max-width: 900px) {
            .terminal-bar {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .platform-header {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str) -> None:
    """Render a consistent participant-facing section heading."""
    st.markdown(
        (
            "<div class='section-shell'>"
            f"<p class='section-kicker'>{html.escape(title)}</p>"
            f"<p class='section-subtitle'>{html.escape(subtitle)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_note_card(text: str, *, quiet: bool = False) -> None:
    """Render a light inline guidance card."""
    quiet_class = " note-card--quiet" if quiet else ""
    st.markdown(
        f"<div class='note-card{quiet_class}'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def render_insight_chips(items: Sequence[tuple[str, str]]) -> None:
    """Render descriptive insight chips."""
    if not items:
        return
    chips_html = "".join(
        f"<span class='insight-chip insight-chip--{html.escape(tone)}'>{html.escape(label)}</span>"
        for label, tone in items
    )
    st.markdown(chips_html, unsafe_allow_html=True)


def render_trade_ticket_banner(*, remaining_slots: int, max_actions_per_step: int) -> None:
    """Render the bold trading-focused banner for the action builder."""
    slot_text = (
        "This week is full. Remove a planned decision before adding another."
        if remaining_slots <= 0
        else f"{remaining_slots} of {max_actions_per_step} action slot(s) still open this week."
    )
    st.markdown(
        (
            "<div class='trade-ticket-shell'>"
            "<p class='trade-ticket-kicker'>Decision Ticket</p>"
            "<p class='trade-ticket-title'>Place your decision for next week’s open</p>"
            "<p class='trade-ticket-subtitle'>"
            "Choose one action, fill in the trade details, and add it to your weekly plan."
            "</p>"
            f"<div class='trade-ticket-status'>{html.escape(slot_text)}</div>"
            "<div>"
            "<span class='trade-step-pill'>1. Choose your action</span>"
            "<span class='trade-step-pill'>2. Choose the stock</span>"
            "<span class='trade-step-pill'>3. Size the trade</span>"
            "<span class='trade-step-pill'>4. Add the decision</span>"
            "</div>"
            "<div class='trade-ticket-note'>"
            "When you submit, any earlier forced sale executes first. The simulator then checks your "
            "plan and carries valid trades to next week’s open."
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_plan_review_banner(*, current_actions: int, max_actions_per_step: int) -> None:
    """Render the review-and-submit banner for the weekly plan."""
    status_text = (
        "No actions in the plan yet."
        if current_actions == 0
        else f"{current_actions} of {max_actions_per_step} action slot(s) currently used."
    )
    st.markdown(
        (
            "<div class='review-ticket-shell'>"
            "<p class='review-ticket-kicker'>Weekly Plan Review</p>"
            "<p class='review-ticket-title'>Review and submit this week’s decisions</p>"
            "<p class='review-ticket-subtitle'>"
            "Check the estimated impact, review the plain-language plan, and submit when it looks right."
            "</p>"
            f"<div class='review-ticket-status'>{html.escape(status_text)}</div>"
            "<div>"
            "<span class='review-step-pill'>5. Review estimated impact</span>"
            "<span class='review-step-pill'>6. Check the plan</span>"
            "<span class='review-step-pill'>7. Submit the week</span>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_session_setup(
    *,
    default_dataset_path: str,
    detected_datasets: Sequence[str],
    default_episode_name: str,
    key_prefix: str = "session_setup",
) -> dict[str, str] | None:
    """Render the initial participant-facing setup form."""
    st.title("Guided Historical Trading Session")
    st.markdown(
        "You will make one weekly investment decision using only the information that would have "
        "been visible at that time."
    )
    render_note_card(
        "The session does not begin at the raw first week of the dataset. Instead, it starts "
        "after enough earlier history has accumulated so your first decision already has context.",
        quiet=True,
    )

    dataset_options = list(detected_datasets)
    default_dataset_index = 0
    if default_dataset_path in dataset_options:
        default_dataset_index = dataset_options.index(default_dataset_path)

    with st.form(f"{key_prefix}_form"):
        participant_id = st.text_input(
            "Participant Code",
            key=f"{key_prefix}_participant_id",
            help="Use a short identifier for this person or pilot session.",
        )
        condition = st.selectbox(
            "Session Type",
            options=list(SUPPORTED_CONDITIONS),
            format_func=condition_display_label,
            key=f"{key_prefix}_condition",
        )
        episode_name = st.text_input(
            "Episode Name",
            value=default_episode_name,
            key=f"{key_prefix}_episode_name",
            help="A short label for this run, for example pilot_01 or participant_07_episode_a.",
        )
        use_custom_dataset = st.checkbox(
            "Use a custom dataset path",
            value=default_dataset_path not in dataset_options,
            key=f"{key_prefix}_use_custom_dataset",
        )

        if use_custom_dataset or not dataset_options:
            dataset_path = st.text_input(
                "Dataset Path",
                value=default_dataset_path,
                key=f"{key_prefix}_dataset_path",
                help="Path to a weekly OHLCV CSV file.",
            )
        else:
            dataset_path = st.selectbox(
                "Choose a Dataset",
                options=dataset_options,
                index=default_dataset_index,
                key=f"{key_prefix}_dataset_select",
            )

        notes = st.text_area(
            "Notes (optional)",
            value="",
            key=f"{key_prefix}_notes",
            help="Optional researcher notes for this local session.",
        )
        submitted = st.form_submit_button("Start Session", type="primary")

    if not submitted:
        return None

    return {
        "participant_id": participant_id,
        "condition": condition,
        "episode_name": episode_name,
        "dataset_path": dataset_path,
        "notes": notes,
    }


def render_session_bar(
    metadata: SessionMetadata,
    observation: Observation,
) -> None:
    """Render the platform-style session summary row."""
    cards = [
        ("Participant", metadata.participant_id),
        ("Condition", condition_display_label(metadata.condition)),
        ("Episode", metadata.episode_name),
        ("Decision Week", f"Week {observation.week_index + 1}"),
        ("Date", observation.date.strftime("%Y-%m-%d")),
    ]
    card_html = "".join(
        (
            "<div class='terminal-bar-card'>"
            f"<div class='terminal-bar-label'>{html.escape(label)}</div>"
            f"<div class='terminal-bar-value'>{html.escape(value)}</div>"
            "</div>"
        )
        for label, value in cards
    )
    st.markdown(f"<div class='terminal-bar'>{card_html}</div>", unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='terminal-bar-meta'>"
            f"Session started at decision week {metadata.decision_start_week + 1} "
            f"with {metadata.visible_history_weeks_at_start} visible week(s) already available."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_market_watchlist_panel(
    observation: Observation,
    *,
    key_prefix: str = "market_watchlist",
) -> str:
    """Render a compact terminal-style watchlist and return the selected ticker."""
    close_history = observation.price_history.loc[:, ["date", "ticker", "close"]].copy()
    previous_closes = _previous_visible_close_lookup(close_history)
    current_week = observation.current_week_ohlcv.copy().sort_values("ticker")
    selection_key = f"{key_prefix}_selected_ticker"
    default_ticker = _default_chart_ticker(observation)

    if (
        selection_key not in st.session_state
        or st.session_state[selection_key] not in observation.available_tickers
    ):
        st.session_state[selection_key] = default_ticker

    st.markdown("<p class='watchlist-title'>Watchlist</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='watchlist-copy'>Latest visible closes and moves versus the previous visible close.</p>",
        unsafe_allow_html=True,
    )

    for row in current_week.itertuples(index=False):
        ticker = str(row.ticker)
        previous_close = previous_closes.get(ticker)
        change_pct = (
            float(row.close) / float(previous_close) - 1.0
            if previous_close is not None and float(previous_close) > _EPSILON
            else None
        )
        row_cols = st.columns([0.45, 0.3, 0.25], gap="small")
        if row_cols[0].button(
            ticker,
            key=f"{key_prefix}_{ticker}",
            type="primary" if st.session_state[selection_key] == ticker else "secondary",
            use_container_width=True,
        ):
            st.session_state[selection_key] = ticker
        row_cols[1].markdown(
            f"<div class='watch-price'>{html.escape(_currency(float(row.close)))}</div>",
            unsafe_allow_html=True,
        )
        row_cols[2].markdown(
            _watchlist_change_badge_html(change_pct),
            unsafe_allow_html=True,
        )

    holdings_count = sum(
        1 for shares in observation.portfolio_state.shares_dict().values() if shares > _EPSILON
    )
    st.markdown(
        (
            "<div class='watch-subcopy'>"
            f"{holdings_count} active holding(s) • "
            f"{len(observation.pending_liquidations)} forced sale(s) already scheduled"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    return str(st.session_state[selection_key])


def render_market_chart_panel(
    observation: Observation,
    *,
    selected_ticker: str,
) -> None:
    """Render the main chart-first market panel for the selected visible ticker."""
    history = observation.price_history.copy()
    history["date"] = pd.to_datetime(history["date"])
    ticker_history = history[history["ticker"] == selected_ticker].sort_values("date").copy()
    current_week = observation.current_week_ohlcv.copy()
    selected_rows = current_week[current_week["ticker"] == selected_ticker]
    selected_row = selected_rows.iloc[0] if not selected_rows.empty else None

    current_close = float(selected_row["close"]) if selected_row is not None else None
    previous_close = (
        float(ticker_history.iloc[-2]["close"])
        if len(ticker_history) >= 2
        else None
    )
    history_weeks_visible = int(ticker_history["date"].nunique()) if not ticker_history.empty else 0

    st.markdown("<p class='panel-title'>Market Context</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='panel-copy'>Only visible history through the current decision week is shown here.</p>",
        unsafe_allow_html=True,
    )

    snapshot_cols = st.columns(5, gap="small")
    snapshot_cols[0].metric("Ticker", selected_ticker)
    snapshot_cols[1].metric(
        "Visible close",
        _currency(current_close) if current_close is not None else "N/A",
    )
    if current_close is not None and previous_close is not None:
        snapshot_cols[1].markdown(
            _price_change_badge_html(current_close - previous_close),
            unsafe_allow_html=True,
        )
    snapshot_cols[2].metric(
        "Week open",
        _currency(float(selected_row["open"])) if selected_row is not None else "N/A",
    )
    snapshot_cols[3].metric(
        "Week range",
        (
            f"{_currency(float(selected_row['low']))} to {_currency(float(selected_row['high']))}"
            if selected_row is not None
            else "N/A"
        ),
    )
    snapshot_cols[4].metric("Visible weeks", str(history_weeks_visible))

    if not ticker_history.empty:
        st.altair_chart(_candlestick_chart(ticker_history), use_container_width=True)
    else:
        render_note_card(
            f"No visible market history is available for {selected_ticker}.",
            quiet=True,
        )
        return

    insight_cols = st.columns([1.2, 0.8], gap="medium")
    with insight_cols[0]:
        render_insight_chips(_market_insight_chips(ticker_history))
        render_note_card(_market_context_summary(ticker_history, selected_ticker), quiet=True)
    with insight_cols[1]:
        st.markdown("**Recent visible close changes**")
        if len(ticker_history) >= 2:
            st.altair_chart(
                _recent_change_chart(ticker_history.tail(min(8, len(ticker_history)))),
                use_container_width=True,
            )
        else:
            render_note_card(
                "Recent weekly changes appear once more than one visible week is available.",
                quiet=True,
            )


def render_market_panel(
    observation: Observation,
    *,
    key_prefix: str = "market_panel",
) -> None:
    """Compatibility renderer for a stacked watchlist and chart view."""
    selected_ticker = render_market_watchlist_panel(observation, key_prefix=f"{key_prefix}_watch")
    render_market_chart_panel(observation, selected_ticker=selected_ticker)


def render_financial_status_panel(state: PortfolioState) -> None:
    """Render prominent summary cards for the participant's current finances."""
    invested_amount = max(0.0, float(state.total_nav - state.cash))
    holdings_count = sum(1 for shares in state.shares_dict().values() if shares > _EPSILON)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Cash", _currency(state.cash))
    summary_cols[1].metric("Invested", _currency(invested_amount))
    summary_cols[2].metric("Account value", _currency(state.total_nav))
    summary_cols[3].metric("Positions", str(holdings_count))
    st.caption("Current cash and invested capital at the end of the visible decision week.")


def render_holdings_panel(state: PortfolioState) -> None:
    """Render the participant's current holdings in a compact readable table."""
    holdings_rows = []
    shares = state.shares_dict()
    market_values = state.market_value_dict()
    cost_basis = state.cost_basis_dict()
    stop_levels = state.stop_levels_dict()

    for ticker in sorted(shares):
        market_value = float(market_values.get(ticker, 0.0))
        weight = market_value / state.total_nav if state.total_nav > _EPSILON else 0.0
        holdings_rows.append(
            {
                "Stock": ticker,
                "Shares Held": _format_shares(shares[ticker]),
                "Average Cost": _currency(cost_basis.get(ticker, 0.0)),
                "Current Market Value": _currency(market_value),
                "Portfolio Weight": _pct(weight),
                "Active Stop": (
                    _currency(stop_levels[ticker]) if ticker in stop_levels else "None"
                ),
            }
        )

    st.markdown("**Open positions**")
    if holdings_rows:
        st.dataframe(pd.DataFrame(holdings_rows), use_container_width=True, hide_index=True)
    else:
        render_note_card(
            "All capital is currently in cash. No stock positions are open right now.",
            quiet=True,
        )


def render_portfolio_insight_panel(state: PortfolioState) -> None:
    """Render participant-facing charts and descriptive portfolio insights."""
    nav_frame = _nav_history_frame(state)
    insight_chips = _portfolio_insight_chips(state)

    top_cols = st.columns([1.25, 0.95], gap="large")
    with top_cols[0]:
        st.markdown("**Account equity so far**")
        if len(nav_frame) >= 2:
            st.altair_chart(
                _portfolio_value_chart(nav_frame, initial_nav=float(state.nav_history[0])),
                use_container_width=True,
            )
        else:
            render_note_card(
                "Your portfolio path will appear here after more weekly steps are completed.",
                quiet=True,
            )

    with top_cols[1]:
        st.markdown("**Portfolio reading**")
        render_insight_chips(insight_chips)
        render_note_card(_portfolio_context_summary(state), quiet=True)
        st.markdown("**Current allocation**")
        st.altair_chart(_allocation_chart(state), use_container_width=True)

    st.markdown("**Drawdown so far**")
    if len(nav_frame) >= 2:
        st.altair_chart(_drawdown_chart(nav_frame), use_container_width=True)
    else:
        render_note_card(
            "Drawdown becomes visible once the portfolio has moved across more than one completed week.",
            quiet=True,
        )


def render_portfolio_panel(state: PortfolioState) -> None:
    """Compatibility wrapper for the portfolio-now view."""
    render_financial_status_panel(state)
    render_holdings_panel(state)


def render_risk_panel(state: PortfolioState) -> None:
    """Render the current portfolio risk snapshot in plain language."""
    risk_cols = st.columns(4)
    risk_cols[0].metric("Realized P/L", _currency(state.realized_pnl))
    risk_cols[1].metric("Concentration", f"{state.concentration_hhi:.4f}")
    risk_cols[2].metric("Turnover", _pct(state.weekly_turnover))
    risk_cols[3].metric(
        "Volatility",
        "Not available yet" if state.portfolio_volatility is None else _pct(state.portfolio_volatility),
    )
    st.caption(
        "Risk controls are enforced by the simulator when later actions are validated."
    )


def render_pending_liquidations_panel(
    pending_liquidations: Sequence[PendingLiquidation],
) -> None:
    """Render any already-scheduled forced sales in plain language."""
    st.markdown("**Pending forced sales**")
    if not pending_liquidations:
        render_note_card(
            "No forced sales are currently scheduled.",
            quiet=True,
        )
        return

    rows = [
        {
            "Stock": item.ticker,
            "Low That Triggered It": _currency(item.triggered_by_low),
            "Stop Price": _currency(item.stop_level),
            "Scheduled Execution Week": item.execution_week + 1,
        }
        for item in sorted(pending_liquidations, key=lambda value: (value.execution_week, value.ticker))
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "These forced sales were already triggered by past price moves. When that scheduled week "
        "begins, they execute before any new trade decisions for that same week."
    )


def render_plan_impact_preview(
    *,
    config: SimulatorConfig,
    observation: Observation,
    current_batch: Sequence[Action],
    max_actions_per_step: int,
) -> None:
    """Render a lightweight plan-impact estimate using only current visible data."""
    preview = _build_plan_impact_preview(
        config=config,
        observation=observation,
        current_batch=current_batch,
    )

    if not current_batch:
        empty_copy = "No new actions added yet. If you submit now, your current portfolio carries into next week."
        if observation.pending_liquidations:
            empty_copy += " Any previously scheduled forced sale would still execute."
        st.markdown(
            (
                "<div class='empty-plan-card'>"
                "<div class='empty-plan-title'>No new actions added yet</div>"
                f"<p class='empty-plan-copy'>{html.escape(empty_copy)}</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    primary_top = st.columns(2, gap="small")
    primary_top[0].metric("Actions in plan", str(len(current_batch)))
    primary_top[1].metric("Est. spend", _currency(preview.estimated_spend))

    primary_bottom = st.columns(2, gap="small")
    primary_bottom[0].metric("Est. costs", _currency(preview.estimated_transaction_costs))
    primary_bottom[1].metric("Est. cash after", _currency(preview.estimated_remaining_cash))

    st.caption("Estimated from currently visible prices only. These are guidance figures, not guaranteed outcomes.")

    secondary_cols = st.columns(3, gap="small")
    secondary_cols[0].metric("Est. proceeds", _currency(preview.estimated_proceeds))
    secondary_cols[1].metric("Est. positions", str(preview.estimated_positions_after))
    secondary_cols[2].metric("Open slots", str(max(0, max_actions_per_step - len(current_batch))))

    supplemental_cols = st.columns(2, gap="small")
    supplemental_cols[0].metric("Est. invested after", _currency(preview.estimated_invested_after))
    supplemental_cols[1].metric(
        "Largest est. weight",
        "N/A" if preview.projected_max_weight is None else _pct(preview.projected_max_weight),
    )

    if preview.warnings:
        _render_feedback_card("Plan warnings", preview.warnings, tone="warning")
    else:
        st.caption("This plan looks broadly feasible from the visible prices, although the simulator will still apply its hard rules when you submit.")

    for note in preview.notes:
        st.caption(note)


def build_plan_impact_preview(
    *,
    config: SimulatorConfig,
    observation: Observation,
    current_batch: Sequence[Action],
) -> PlanImpactPreview:
    """Return the participant-facing plan preview using only visible data."""
    return _build_plan_impact_preview(
        config=config,
        observation=observation,
        current_batch=current_batch,
    )


def build_trade_planner_props(
    *,
    config: SimulatorConfig,
    observation: Observation,
    current_batch: Sequence[Action],
) -> dict[str, Any]:
    """Build a decision-time-only payload for the TypeScript trade planner."""
    state = observation.portfolio_state
    shares = state.shares_dict()
    market_values = state.market_value_dict()
    cost_basis = state.cost_basis_dict()
    stop_levels = state.stop_levels_dict()
    close_prices = _close_price_lookup(observation)
    preview = build_plan_impact_preview(
        config=config,
        observation=observation,
        current_batch=current_batch,
    )

    holdings_payload = [
        {
            "ticker": ticker,
            "shares": float(shares[ticker]),
            "average_cost": float(cost_basis.get(ticker, 0.0)),
            "market_value": float(market_values.get(ticker, 0.0)),
            "weight": (
                float(market_values.get(ticker, 0.0)) / float(state.total_nav)
                if state.total_nav > _EPSILON
                else 0.0
            ),
            "active_stop": (
                float(stop_levels[ticker]) if ticker in stop_levels else None
            ),
            "current_close": float(close_prices.get(ticker, 0.0)) if ticker in close_prices else None,
        }
        for ticker in sorted(shares)
    ]

    ticker_options = {
        action_type.value: _ticker_options_for_action(
            action_type=action_type,
            available_tickers=observation.available_tickers,
            holdings=shares,
            active_stop_tickers=stop_levels.keys(),
        )
        for action_type in (
            ActionType.BUY,
            ActionType.SELL,
            ActionType.REDUCE,
            ActionType.SET_STOP,
            ActionType.REMOVE_STOP,
        )
    }

    current_batch_payload = [
        {
            "action_type": action.action_type.value,
            "ticker": action.ticker,
            "quantity": float(action.quantity) if action.quantity is not None else None,
            "quantity_type": (
                action.quantity_type.value if action.quantity_type is not None else None
            ),
            "fraction": float(action.fraction) if action.fraction is not None else None,
            "stop_price": float(action.stop_price) if action.stop_price is not None else None,
            "summary": _action_summary(action),
            "detail": _action_detail(action),
        }
        for action in current_batch
    ]

    return {
        "current_week_index": int(observation.week_index),
        "current_date": observation.date.isoformat(),
        "max_actions_per_step": int(config.max_actions_per_step),
        "remaining_action_slots": max(0, config.max_actions_per_step - len(current_batch)),
        "available_tickers": list(observation.available_tickers),
        "current_cash": float(state.cash),
        "current_total_nav": float(state.total_nav),
        "current_batch": current_batch_payload,
        "holdings": holdings_payload,
        "active_stops": {ticker: float(value) for ticker, value in sorted(stop_levels.items())},
        "pending_liquidations": [
            {
                "ticker": item.ticker,
                "triggered_by_low": float(item.triggered_by_low),
                "stop_level": float(item.stop_level),
                "execution_week": int(item.execution_week),
            }
            for item in sorted(
                observation.pending_liquidations,
                key=lambda value: (value.execution_week, value.ticker),
            )
        ],
        "ticker_options": ticker_options,
        "close_prices": {ticker: float(value) for ticker, value in sorted(close_prices.items())},
        "plan_impact": {
            "estimated_spend": float(preview.estimated_spend),
            "estimated_proceeds": float(preview.estimated_proceeds),
            "estimated_transaction_costs": float(preview.estimated_transaction_costs),
            "estimated_remaining_cash": float(preview.estimated_remaining_cash),
            "estimated_positions_after": int(preview.estimated_positions_after),
            "estimated_invested_after": float(preview.estimated_invested_after),
            "projected_max_weight": (
                float(preview.projected_max_weight)
                if preview.projected_max_weight is not None
                else None
            ),
            "warnings": list(preview.warnings),
            "notes": list(preview.notes),
        },
    }


def render_action_builder(
    *,
    config: SimulatorConfig,
    observation: Observation,
    current_batch: Sequence[Action],
    key_prefix: str = "action_builder",
) -> tuple[Action | None, str | None]:
    """Render one guided action builder with participant-friendly language."""
    remaining_slots = config.max_actions_per_step - len(current_batch)
    if remaining_slots <= 0:
        _render_feedback_card(
            "This week's plan is already full",
            ("Remove a planned decision before adding another one.",),
            tone="warning",
        )
        return None, None

    builder_choices = [
        ActionType.BUY,
        ActionType.SELL,
        ActionType.REDUCE,
        ActionType.SET_STOP,
        ActionType.REMOVE_STOP,
    ]
    st.markdown("**Step 1 · Choose your action**")
    selected_action_type = _action_type_from_label(
        st.selectbox(
            "Choose your action",
            options=[_ACTION_LABELS[action_type] for action_type in builder_choices],
            key=f"{key_prefix}_action_type_label",
        )
    )
    st.caption(_ACTION_HELP_TEXT[selected_action_type])

    ticker: str | None = None
    quantity: float | None = None
    quantity_type: QuantityType | None = None
    fraction: float | None = None
    stop_price: float | None = None

    holdings = observation.portfolio_state.shares_dict()
    active_stops = observation.portfolio_state.stop_levels_dict()
    close_prices = _close_price_lookup(observation)
    ticker_options = _ticker_options_for_action(
        action_type=selected_action_type,
        available_tickers=observation.available_tickers,
        holdings=holdings,
        active_stop_tickers=active_stops.keys(),
    )

    can_add = True
    if selected_action_type != ActionType.HOLD:
        if not ticker_options:
            can_add = False
            _render_feedback_card(
                "This action is not available right now",
                (_no_ticker_message(selected_action_type),),
                tone="warning",
            )
        else:
            st.markdown("**Step 2 · Choose the stock**")
            ticker = st.selectbox(
                "Choose a stock",
                options=ticker_options,
                key=f"{key_prefix}_ticker",
            )

    if selected_action_type == ActionType.BUY:
        st.markdown("**Step 3 · Choose how to size this trade**")
        quantity_type = QuantityType(
            st.selectbox(
                "How would you like to size this purchase?",
                options=[
                    QuantityType.SHARES.value,
                    QuantityType.NOTIONAL_DOLLARS.value,
                    QuantityType.NAV_FRACTION.value,
                ],
                format_func=_quantity_type_label,
                key=f"{key_prefix}_buy_quantity_type",
            )
        )
        st.markdown("**Step 4 · Enter the amount**")
        quantity = st.number_input(
            _quantity_input_label(quantity_type),
            min_value=0.0001,
            value=1.0,
            step=0.01,
            key=f"{key_prefix}_buy_quantity",
        )
    elif selected_action_type == ActionType.SELL:
        if holdings:
            st.caption("Stocks you currently own are shown first to make selling easier.")
        else:
            st.caption("You do not currently own any stocks. A sell request is likely to be rejected.")
        st.markdown("**Step 3 · Choose how to size this trade**")
        quantity_type = QuantityType(
            st.selectbox(
                "How would you like to size this sale?",
                options=[
                    QuantityType.SHARES.value,
                    QuantityType.NOTIONAL_DOLLARS.value,
                    QuantityType.CLOSE_ALL.value,
                ],
                format_func=_quantity_type_label,
                key=f"{key_prefix}_sell_quantity_type",
            )
        )
        if quantity_type != QuantityType.CLOSE_ALL:
            st.markdown("**Step 4 · Enter the amount**")
            quantity = st.number_input(
                _quantity_input_label(quantity_type),
                min_value=0.0001,
                value=1.0,
                step=0.01,
                key=f"{key_prefix}_sell_quantity",
            )
    elif selected_action_type == ActionType.REDUCE:
        st.markdown("**Step 3 · Enter the amount to reduce**")
        fraction = st.number_input(
            "What share of this holding would you like to reduce? (0.25 = 25%)",
            min_value=0.0001,
            max_value=1.0,
            value=0.25,
            step=0.01,
            key=f"{key_prefix}_reduce_fraction",
        )
    elif selected_action_type == ActionType.SET_STOP:
        default_stop = 1.0
        if ticker is not None and ticker in close_prices:
            default_stop = round(close_prices[ticker] * 0.90, 2)
            lower_bound = close_prices[ticker] * (1.0 - config.stop_max_pct)
            upper_bound = close_prices[ticker] * (1.0 - config.stop_min_pct)
            st.caption(
                f"Current close: {_currency(close_prices[ticker])}. A typical valid stop range for this "
                f"week is about {_currency(lower_bound)} to {_currency(upper_bound)}."
            )
        st.markdown("**Step 3 · Enter the stop level**")
        stop_price = st.number_input(
            "Enter the stop price",
            min_value=0.0001,
            value=default_stop,
            step=0.01,
            key=f"{key_prefix}_stop_price",
        )

    add_clicked = st.button(
        "Add this decision to your weekly plan",
        type="secondary",
        key=f"{key_prefix}_add_button",
        disabled=not can_add,
        use_container_width=True,
    )
    if not add_clicked:
        return None, None

    try:
        action = Action(
            action_type=selected_action_type,
            ticker=ticker,
            quantity=quantity,
            quantity_type=quantity_type,
            fraction=fraction,
            stop_price=stop_price,
        )
    except (TypeError, ValueError) as exc:
        return None, f"Could not add this decision: {exc}"

    return action, None


def render_action_batch_preview(
    current_batch: Sequence[Action],
    *,
    max_actions_per_step: int,
    pending_forced_sales_count: int = 0,
    key_prefix: str = "batch_preview",
) -> dict[str, Any]:
    """Render a plain-language summary of the participant's planned actions."""
    remove_index: int | None = None
    if current_batch:
        st.markdown("**Plan summary**")
        for index, action in enumerate(current_batch):
            row_cols = st.columns([0.79, 0.21], gap="small")
            detail = _action_detail(action)
            detail_html = (
                f"<div class='plan-row-caption'>{html.escape(detail)}</div>"
                if detail
                else ""
            )
            row_cols[0].markdown(
                (
                    "<div class='plan-row-card'>"
                    f"<div class='plan-row-index'>Decision {index + 1}</div>"
                    f"<div class='plan-row-title'>{html.escape(_action_summary(action))}</div>"
                    f"{detail_html}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if row_cols[1].button(
                "Remove",
                key=f"{key_prefix}_remove_{index}",
                use_container_width=True,
            ):
                remove_index = index
    elif pending_forced_sales_count > 0:
        st.caption("No new actions are in the plan. Earlier forced sales would still execute if already scheduled.")

    st.caption("Earlier forced sales execute first. Any new trades that pass the simulator rules then move to next week’s open.")

    st.markdown("**Ready to submit**")
    controls = st.columns([0.55, 0.45], gap="small")
    clear_clicked = controls[0].button(
        "Clear this plan",
        key=f"{key_prefix}_clear_button",
        disabled=not current_batch,
        use_container_width=True,
    )
    submit_clicked = controls[1].button(
        "Submit this week's decisions",
        type="primary",
        key=f"{key_prefix}_submit_button",
        use_container_width=True,
    )
    return {
        "remove_index": remove_index,
        "clear_batch": clear_clicked,
        "submit_batch": submit_clicked,
    }


def render_step_feedback(step_info: dict[str, Any] | None) -> None:
    """Render participant-friendly feedback after the most recent step."""
    if not step_info:
        return

    render_section_header(
        "What changed after your last decision",
        "A concise view of how the simulator handled the previous weekly plan.",
    )

    summary_cols = st.columns(4)
    cash_before = float(step_info.get("cash_before", 0.0))
    cash_after = float(step_info.get("cash_after", 0.0))
    invested_before = float(step_info.get("invested_before", 0.0))
    invested_after = float(step_info.get("invested_after", 0.0))
    total_nav_before = float(step_info.get("total_nav_before", 0.0))
    total_nav_after = float(step_info.get("total_nav_after", step_info.get("batch_end_nav", 0.0)))
    holdings_before = int(step_info.get("holdings_before_count", 0))
    holdings_after = int(step_info.get("holdings_after_count", 0))

    summary_cols[0].metric(
        "Cash available now",
        _currency(cash_after),
        delta=_signed_currency_text(cash_after - cash_before),
    )
    summary_cols[0].caption(f"Before: {_currency(cash_before)}")
    summary_cols[1].metric(
        "Currently invested",
        _currency(invested_after),
        delta=_signed_currency_text(invested_after - invested_before),
    )
    summary_cols[1].caption(f"Before: {_currency(invested_before)}")
    summary_cols[2].metric(
        "Total portfolio value",
        _currency(total_nav_after),
        delta=_signed_currency_text(total_nav_after - total_nav_before),
    )
    summary_cols[2].caption(f"Before: {_currency(total_nav_before)}")
    summary_cols[3].metric(
        "Number of holdings",
        str(holdings_after),
        delta=_signed_number_text(holdings_after - holdings_before),
    )
    summary_cols[3].caption(f"Before: {holdings_before}")

    if step_info.get("n_actions_submitted", 0) == 0:
        render_note_card(
            "No new action was submitted last week, so the portfolio simply carried forward into the next week.",
            quiet=True,
        )

    position_change_items = tuple(step_info.get("position_change_items", ()))
    if position_change_items:
        _render_feedback_card("Position changes", position_change_items, tone="success")

    outcome_cols = st.columns(3)
    outcome_cols[0].metric("Accepted actions", int(step_info.get("n_accepted", 0)))
    outcome_cols[1].metric("Adjusted actions", int(step_info.get("n_clipped", 0)))
    outcome_cols[2].metric("Rejected actions", int(step_info.get("n_rejected", 0)))

    accepted_items: list[str] = []
    adjusted_items: list[str] = []
    rejected_items: list[str] = []

    for result in step_info.get("validation_results", ()):
        if not isinstance(result, ValidationResult):
            continue
        original_summary = _action_summary(result.original_action)
        effective_summary = _action_summary(result.effective_action) if result.effective_action else None

        if result.outcome.value == "accepted":
            accepted_items.append(original_summary)
        elif result.outcome.value == "clipped":
            adjusted_text = original_summary
            if effective_summary and effective_summary != original_summary:
                adjusted_text = f"{original_summary} -> adjusted to {effective_summary}"
            if result.reason:
                adjusted_text = f"{adjusted_text}. Reason: {_humanize_reason(result.reason)}"
            adjusted_items.append(adjusted_text)
        else:
            rejected_text = original_summary
            if result.reason:
                rejected_text = f"{rejected_text}. Reason: {_humanize_reason(result.reason)}"
            rejected_items.append(rejected_text)

    action_groups = [
        ("Accepted decisions", accepted_items, "success"),
        ("Adjusted decisions", adjusted_items, "warning"),
        ("Rejected decisions", rejected_items, "error"),
    ]
    populated_action_groups = [group for group in action_groups if group[1]]
    if populated_action_groups:
        cols = st.columns(len(populated_action_groups))
        for col, (title, items, tone) in zip(cols, populated_action_groups):
            with col:
                _render_feedback_card(title, items, tone=tone)

    forced_sale_items = [
        (
            f"A previously scheduled forced sale executed for {entry.ticker} at "
            f"{_currency(entry.execution_price)}."
        )
        for entry in step_info.get("pending_liquidation_executions", ())
    ]
    stop_trigger_items = [
        (
            f"{entry.ticker} fell to {_currency(entry.triggered_by_low)} and breached its stop price of "
            f"{_currency(entry.stop_level)}."
            + (
                f" A forced sale is now scheduled for week {entry.execution_week + 1}."
                if entry.execution_week is not None
                else " No later week was available to schedule a forced sale."
            )
        )
        for entry in step_info.get("stop_triggers", ())
    ]
    gap_items = [
        (
            f"{entry.ticker} buy size was reduced from {entry.original_shares:.4f} shares to "
            f"{entry.adjusted_shares:.4f} shares because next week's open would have required more cash."
        )
        for entry in step_info.get("gap_adjustments", ())
    ]
    zero_size_skip_items = [str(item) for item in step_info.get("execution_skips", ())]
    event_groups = [
        ("Forced sales that executed", forced_sale_items, "warning"),
        ("New forced sales scheduled", stop_trigger_items, "warning"),
        ("Skipped trades", zero_size_skip_items, "warning"),
        ("Execution-time adjustments", gap_items, "neutral"),
    ]
    populated_event_groups = [group for group in event_groups if group[1]]
    if populated_event_groups:
        cols = st.columns(len(populated_event_groups))
        for col, (title, items, tone) in zip(cols, populated_event_groups):
            with col:
                _render_feedback_card(title, items, tone=tone)

    skipped_stop_actions = step_info.get("skipped_stop_actions", ())
    if skipped_stop_actions:
        render_note_card(
            "Some stop changes could not be applied because trade execution changed the live holdings.",
            quiet=True,
        )

    termination_reason = step_info.get("termination_reason")
    if termination_reason:
        render_note_card(_termination_reason_message(termination_reason), quiet=True)


def render_final_summary(
    metadata: SessionMetadata,
    state: PortfolioState,
    metrics: SimulationMetrics,
    *,
    export_path: str | None = None,
) -> None:
    """Render the participant-facing end-of-session summary."""
    st.header("Session complete")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Final portfolio value", _currency(state.total_nav))
    summary_cols[1].metric("Total return", _pct(metrics.total_return))
    summary_cols[2].metric("Largest drawdown", _pct(metrics.max_drawdown))

    secondary_cols = st.columns(4)
    secondary_cols[0].metric("Realized volatility", _pct(metrics.realized_vol))
    secondary_cols[1].metric("Average weekly turnover", _pct(metrics.avg_weekly_turnover))
    secondary_cols[2].metric("Average concentration", f"{metrics.avg_hhi:.4f}")
    secondary_cols[3].metric("Blow-up flag", "Yes" if metrics.blow_up_flag else "No")

    st.markdown("**Session details**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Field": "Participant code", "Value": metadata.participant_id},
                {"Field": "Session type", "Value": condition_display_label(metadata.condition)},
                {"Field": "Episode name", "Value": metadata.episode_name},
                {"Field": "Dataset path", "Value": metadata.dataset_path},
                {"Field": "First decision week", "Value": metadata.decision_start_week + 1},
                {
                    "Field": "Visible history at start",
                    "Value": f"{metadata.visible_history_weeks_at_start} week(s)",
                },
                {"Field": "Started at", "Value": metadata.started_at.isoformat()},
                {
                    "Field": "Finished at",
                    "Value": metadata.finished_at.isoformat() if metadata.finished_at else "Not finished",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Research metrics**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Metric": "Sharpe ratio", "Value": metrics.sharpe_ratio},
                {"Metric": "Invalid attempts", "Value": metrics.n_invalid_attempts},
                {"Metric": "Adjusted trades", "Value": metrics.n_clipped_trades},
                {"Metric": "Stop triggers", "Value": metrics.n_stop_triggers},
                {"Metric": "Gap adjustments", "Value": metrics.n_gap_adjustments},
                {
                    "Metric": "Volatility rule activation week",
                    "Value": (
                        "Never activated"
                        if metrics.vol_rule_activation_week is None
                        else metrics.vol_rule_activation_week + 1
                    ),
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if export_path:
        st.success(f"Session files were written to: {export_path}")


def render_coach_placeholder(condition: str) -> None:
    """Render the future coach area without cluttering the core decision flow."""
    if condition != "human_with_coach_placeholder":
        return
    st.markdown("**Coach panel**")
    render_note_card(
        "AI coach not connected yet. This space is reserved for future support.",
        quiet=True,
    )


def _ticker_options_for_action(
    *,
    action_type: ActionType,
    available_tickers: Sequence[str],
    holdings: dict[str, float],
    active_stop_tickers: Sequence[str],
) -> list[str]:
    held_tickers = sorted(holdings.keys())
    stop_tickers = sorted(active_stop_tickers)

    if action_type == ActionType.BUY:
        return list(available_tickers)
    if action_type == ActionType.SELL:
        return held_tickers if held_tickers else list(available_tickers)
    if action_type in {ActionType.REDUCE, ActionType.SET_STOP}:
        return held_tickers
    if action_type == ActionType.REMOVE_STOP:
        return stop_tickers
    return list(available_tickers)


def _no_ticker_message(action_type: ActionType) -> str:
    if action_type == ActionType.REDUCE:
        return "You do not currently hold any stocks that can be reduced."
    if action_type == ActionType.SET_STOP:
        return "You need an active holding before you can set a stop price."
    if action_type == ActionType.REMOVE_STOP:
        return "You do not currently have any active stop prices to remove."
    return "No stocks are available for this action."


def _quantity_type_label(value: str) -> str:
    if value == QuantityType.SHARES.value:
        return "By number of shares"
    if value == QuantityType.NOTIONAL_DOLLARS.value:
        return "By dollar amount"
    if value == QuantityType.NAV_FRACTION.value:
        return "By fraction of portfolio value"
    if value == QuantityType.CLOSE_ALL.value:
        return "Sell the entire holding"
    return value.replace("_", " ").title()


def _quantity_input_label(quantity_type: QuantityType) -> str:
    if quantity_type == QuantityType.SHARES:
        return "How many shares?"
    if quantity_type == QuantityType.NOTIONAL_DOLLARS:
        return "How many dollars?"
    if quantity_type == QuantityType.NAV_FRACTION:
        return "What fraction of your portfolio value? (0.10 = 10%)"
    return "Amount"


def _action_type_from_label(label: str) -> ActionType:
    for action_type, display_label in _ACTION_LABELS.items():
        if display_label == label:
            return action_type
    raise ValueError(f"Unknown action label: {label}")


def _close_price_lookup(observation: Observation) -> dict[str, float]:
    return {
        str(row["ticker"]): float(row["close"])
        for _, row in observation.current_week_ohlcv.iterrows()
    }


def _build_plan_impact_preview(
    *,
    config: SimulatorConfig,
    observation: Observation,
    current_batch: Sequence[Action],
) -> PlanImpactPreview:
    state = observation.portfolio_state
    close_prices = _close_price_lookup(observation)
    adv_lookup = _adv_shares_lookup(observation, config.adv_lookback_weeks)
    batch_start_nav = float(state.total_nav)

    projected_cash = float(state.cash)
    projected_shares = dict(state.shares_dict())
    stop_levels = dict(state.stop_levels_dict())

    estimated_spend = 0.0
    estimated_proceeds = 0.0
    estimated_costs = 0.0
    warnings: list[str] = []
    notes: list[str] = []

    ordered_batch = sorted(
        list(current_batch),
        key=lambda action: (_PREVIEW_ACTION_ORDER[action.action_type], _action_summary(action)),
    )

    if not ordered_batch:
        notes.append("No new decision is currently in the plan, so no new cash use is expected.")

    for action in ordered_batch:
        if action.action_type == ActionType.HOLD:
            notes.append("Choosing to do nothing this week does not change your cash or holdings.")
            continue

        ticker = action.ticker
        if ticker is None:
            warnings.append("One planned action is incomplete and may be rejected.")
            continue

        if action.action_type in {ActionType.SET_STOP, ActionType.REMOVE_STOP}:
            if action.action_type == ActionType.SET_STOP:
                stop_levels[ticker] = float(action.stop_price or 0.0)
                notes.append(f"Setting a stop on {ticker} does not use cash by itself.")
            else:
                stop_levels.pop(ticker, None)
                notes.append(f"Removing a stop from {ticker} does not use cash by itself.")
            continue

        reference_price = close_prices.get(ticker)
        if reference_price is None or reference_price <= 0.0:
            warnings.append(
                f"{ticker} cannot be estimated from the currently visible prices and may be treated differently by the simulator."
            )
            continue

        current_shares = float(projected_shares.get(ticker, 0.0))
        signed_shares = _resolve_preview_shares(
            action=action,
            reference_price=float(reference_price),
            shares_held=current_shares,
            batch_start_nav=batch_start_nav,
        )

        if abs(signed_shares) <= _EPSILON:
            if action.action_type in {ActionType.SELL, ActionType.REDUCE}:
                warnings.append(
                    f"{ticker} is not currently held in this estimate, so that sell or reduce action may be rejected."
                )
            continue

        gross_trade_value = abs(signed_shares) * float(reference_price)
        trade_cost = _estimate_trade_cost(
            config=config,
            ticker=ticker,
            reference_price=float(reference_price),
            signed_shares=signed_shares,
            gross_trade_value=gross_trade_value,
            adv_lookup=adv_lookup,
        )
        estimated_costs += trade_cost

        if signed_shares > 0.0:
            estimated_spend += gross_trade_value
            projected_cash -= gross_trade_value + trade_cost
        else:
            estimated_proceeds += gross_trade_value
            projected_cash += gross_trade_value - trade_cost

        new_shares = max(0.0, current_shares + signed_shares)
        if new_shares <= _EPSILON:
            projected_shares.pop(ticker, None)
            stop_levels.pop(ticker, None)
        else:
            projected_shares[ticker] = float(new_shares)

    projected_market_values = {
        ticker: float(shares * close_prices[ticker])
        for ticker, shares in projected_shares.items()
        if shares > _EPSILON and ticker in close_prices
    }
    projected_nav = float(projected_cash + sum(projected_market_values.values()))
    estimated_invested_after = max(0.0, projected_nav - projected_cash)
    estimated_positions_after = sum(1 for shares in projected_shares.values() if shares > _EPSILON)
    projected_max_weight = None
    if projected_nav > _EPSILON and projected_market_values:
        projected_max_weight = max(projected_market_values.values()) / projected_nav

    if estimated_spend > 0.0 and state.cash > _EPSILON:
        if projected_cash < 0.0:
            warnings.append(
                "This plan appears to use more cash than you currently have and may be reduced or rejected by simulator rules."
            )
        elif projected_cash < state.cash * 0.10:
            warnings.append("This plan may use most of your available cash.")

    if projected_max_weight is not None:
        if projected_max_weight > config.single_stock_cap:
            warnings.append(
                "This plan may create a very concentrated portfolio and could be reduced by the simulator's risk rules."
            )
        elif projected_max_weight > 0.35:
            warnings.append("This plan may increase concentration in one stock.")

    unique_notes = tuple(dict.fromkeys(note for note in notes if note))
    unique_warnings = tuple(dict.fromkeys(warning for warning in warnings if warning))
    return PlanImpactPreview(
        estimated_spend=float(estimated_spend),
        estimated_proceeds=float(estimated_proceeds),
        estimated_transaction_costs=float(estimated_costs),
        estimated_remaining_cash=float(projected_cash),
        estimated_positions_after=int(estimated_positions_after),
        estimated_invested_after=float(estimated_invested_after),
        projected_max_weight=projected_max_weight,
        warnings=unique_warnings,
        notes=unique_notes,
    )


def _resolve_preview_shares(
    *,
    action: Action,
    reference_price: float,
    shares_held: float,
    batch_start_nav: float,
) -> float:
    if action.action_type == ActionType.BUY:
        assert action.quantity_type is not None
        assert action.quantity is not None
        if action.quantity_type == QuantityType.SHARES:
            return float(action.quantity)
        if action.quantity_type == QuantityType.NOTIONAL_DOLLARS:
            return float(action.quantity) / reference_price
        if action.quantity_type == QuantityType.NAV_FRACTION:
            return (float(action.quantity) * batch_start_nav) / reference_price
        return 0.0

    if action.action_type == ActionType.SELL:
        assert action.quantity_type is not None
        if action.quantity_type == QuantityType.CLOSE_ALL:
            return -float(shares_held)
        assert action.quantity is not None
        if action.quantity_type == QuantityType.SHARES:
            return -float(action.quantity)
        if action.quantity_type == QuantityType.NOTIONAL_DOLLARS:
            return -(float(action.quantity) / reference_price)
        return 0.0

    if action.action_type == ActionType.REDUCE:
        assert action.fraction is not None
        return -(float(shares_held) * float(action.fraction))

    return 0.0


def _adv_shares_lookup(observation: Observation, adv_lookback_weeks: int) -> dict[str, float]:
    history = observation.price_history.copy()
    if history.empty:
        return {}

    history = history.sort_values(["ticker", "_week_idx"])
    lookup: dict[str, float] = {}
    for ticker, group in history.groupby("ticker", sort=True):
        recent = group.tail(adv_lookback_weeks)
        if recent.empty:
            continue
        lookup[str(ticker)] = float(pd.to_numeric(recent["volume"], errors="coerce").mean())
    return lookup


def _estimate_trade_cost(
    *,
    config: SimulatorConfig,
    ticker: str,
    reference_price: float,
    signed_shares: float,
    gross_trade_value: float,
    adv_lookup: dict[str, float],
) -> float:
    del signed_shares
    adv_shares = float(adv_lookup.get(ticker, 0.0))
    adv_value = adv_shares * reference_price
    if adv_value <= 0.0:
        adv_value = max(gross_trade_value, reference_price)
    impact_bps = 0.0 if gross_trade_value <= 0.0 else (
        (gross_trade_value / adv_value) * config.impact_factor * 10_000.0
    )
    slippage_bps = config.base_slippage_bps + impact_bps
    return gross_trade_value * (
        config.commission_rate + config.spread_rate + (slippage_bps / 10_000.0)
    )


def _previous_visible_close_lookup(close_history: pd.DataFrame) -> dict[str, float]:
    history = close_history.sort_values(["ticker", "date"]).copy()
    previous: dict[str, float] = {}
    for ticker, group in history.groupby("ticker", sort=True):
        if len(group) >= 2:
            previous[str(ticker)] = float(group.iloc[-2]["close"])
    return previous


def _configure_dark_chart(
    chart: alt.Chart | alt.LayerChart | alt.ConcatChart,
) -> alt.Chart | alt.LayerChart | alt.ConcatChart:
    """Apply a dark trading-terminal style to Altair charts."""
    return (
        chart.configure(background="transparent")
        .configure_view(strokeOpacity=0, fill="#131b25")
        .configure_axis(
            labelColor="#8ea6bc",
            titleColor="#8ea6bc",
            gridColor="#223140",
            domainColor="#314455",
            tickColor="#314455",
        )
        .configure_legend(
            labelColor="#c8d5e0",
            titleColor="#9cb0c3",
            strokeColor="#243548",
            fillColor="#131b25",
        )
        .configure_title(color="#eff5fb")
    )


def _market_history_chart(ticker_history: pd.DataFrame, ticker: str) -> alt.Chart:
    history = ticker_history.copy()
    history["date"] = pd.to_datetime(history["date"])
    base = (
        alt.Chart(history)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("close:Q", title="Visible Close"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("close:Q", title="Close", format=",.2f"),
            ],
        )
    )
    line = base.mark_line(color="#4da3ff", strokeWidth=2.4)
    points = base.mark_circle(color="#9bd0ff", size=38)
    return _configure_dark_chart((line + points).properties(height=280))


def _candlestick_chart(ticker_history: pd.DataFrame) -> alt.ConcatChart:
    """Render a dark OHLC chart with a compact volume panel."""
    history = ticker_history.copy().sort_values("date")
    history["date"] = pd.to_datetime(history["date"])
    history["direction"] = history.apply(
        lambda row: "Up" if float(row["close"]) >= float(row["open"]) else "Down",
        axis=1,
    )
    history["body_low"] = history[["open", "close"]].min(axis=1)
    history["body_high"] = history[["open", "close"]].max(axis=1)

    price_base = alt.Chart(history).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(labelAngle=0)),
        tooltip=[
            alt.Tooltip("date:T", title="Week"),
            alt.Tooltip("open:Q", title="Open", format=",.2f"),
            alt.Tooltip("high:Q", title="High", format=",.2f"),
            alt.Tooltip("low:Q", title="Low", format=",.2f"),
            alt.Tooltip("close:Q", title="Close", format=",.2f"),
            alt.Tooltip("volume:Q", title="Volume", format=",.0f"),
        ],
    )
    wicks = price_base.mark_rule(color="#98a7b6").encode(
        y=alt.Y("low:Q", title="Price"),
        y2="high:Q",
    )
    bodies = price_base.mark_bar(size=10).encode(
        y="body_low:Q",
        y2="body_high:Q",
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["Up", "Down"], range=["#1f8f5f", "#dc5d5d"]),
            legend=None,
        ),
    )
    price_chart = (wicks + bodies).properties(height=390)

    volume_chart = (
        alt.Chart(history)
        .mark_bar(opacity=0.72)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("volume:Q", title="Volume"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Up", "Down"], range=["#1f8f5f", "#dc5d5d"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Week"),
                alt.Tooltip("volume:Q", title="Volume", format=",.0f"),
            ],
        )
        .properties(height=100)
    )
    return _configure_dark_chart(
        alt.vconcat(price_chart, volume_chart, spacing=8).resolve_scale(x="shared")
    )


def _recent_change_chart(recent_slice: pd.DataFrame) -> alt.Chart:
    history = recent_slice.copy()
    history["date"] = pd.to_datetime(history["date"])
    history["weekly_change"] = history["close"].pct_change().fillna(0.0)
    history["direction"] = history["weekly_change"].apply(
        lambda value: "Positive" if value > _EPSILON else ("Negative" if value < -_EPSILON else "Flat")
    )
    return _configure_dark_chart(
        alt.Chart(history)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("date:T", title="Week"),
            y=alt.Y("weekly_change:Q", title="Weekly Close Change"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Positive", "Negative", "Flat"],
                    range=["#1f8f5f", "#dc5d5d", "#607a96"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Week"),
                alt.Tooltip("weekly_change:Q", title="Weekly Change", format=".2%"),
            ],
        )
        .properties(height=190)
    )


def _market_insight_chips(ticker_history: pd.DataFrame) -> list[tuple[str, str]]:
    history = ticker_history.sort_values("date").copy()
    if len(history) < 2:
        return [("Limited visible history so far", "neutral")]

    window = history.tail(min(10, len(history)))
    closes = pd.to_numeric(window["close"], errors="coerce")
    trend_return = float(closes.iloc[-1] / closes.iloc[0] - 1.0) if closes.iloc[0] > 0 else 0.0
    returns = closes.pct_change().dropna()
    realized_vol = float(returns.std()) if not returns.empty else 0.0

    chips: list[tuple[str, str]] = []
    if trend_return > 0.06:
        chips.append(("Recent visible trend is upward", "good"))
    elif trend_return < -0.06:
        chips.append(("Recent visible trend is downward", "risk"))
    else:
        chips.append(("Recent visible trend is relatively flat", "neutral"))

    if realized_vol > 0.05:
        chips.append(("Recent history has been volatile", "warn"))
    elif len(returns) >= 3 and realized_vol < 0.02:
        chips.append(("Recent history has been relatively stable", "good"))

    max_close = float(closes.max())
    min_close = float(closes.min())
    current_close = float(closes.iloc[-1])
    if max_close > 0 and current_close >= max_close * 0.98:
        chips.append(("Current close is near the visible high", "good"))
    elif min_close > 0 and current_close <= min_close * 1.02:
        chips.append(("Current close is near the visible low", "risk"))
    return chips


def _market_context_summary(ticker_history: pd.DataFrame, ticker: str) -> str:
    history = ticker_history.sort_values("date").copy()
    if len(history) < 2:
        return f"{ticker} has only a short visible history so far, so the current reading is still limited."

    window = history.tail(min(10, len(history)))
    closes = pd.to_numeric(window["close"], errors="coerce")
    trend_return = float(closes.iloc[-1] / closes.iloc[0] - 1.0) if closes.iloc[0] > 0 else 0.0
    direction = "upward" if trend_return > 0.03 else ("downward" if trend_return < -0.03 else "mostly sideways")
    return (
        f"Over the recent visible window, {ticker} has moved in a {direction} pattern. "
        "This is descriptive context only, not a forecast."
    )


def _nav_history_frame(state: PortfolioState) -> pd.DataFrame:
    nav_values = list(state.nav_history)
    start_week = state.week_index - len(nav_values) + 1
    rows: list[dict[str, float | int]] = []
    running_peak = 0.0
    for offset, nav in enumerate(nav_values):
        week_index = start_week + offset
        running_peak = max(running_peak, float(nav))
        drawdown = (float(nav) / running_peak - 1.0) if running_peak > 0.0 else 0.0
        rows.append(
            {
                "decision_step": offset + 1,
                "week_number": week_index + 1,
                "nav": float(nav),
                "drawdown": float(drawdown),
            }
        )
    return pd.DataFrame(rows)


def _portfolio_value_chart(nav_frame: pd.DataFrame, initial_nav: float) -> alt.Chart:
    base = alt.Chart(nav_frame).encode(
        x=alt.X("decision_step:Q", title="Decision Step"),
        y=alt.Y("nav:Q", title="Portfolio Value"),
        tooltip=[
            alt.Tooltip("week_number:Q", title="Week"),
            alt.Tooltip("nav:Q", title="Portfolio Value", format=",.2f"),
        ],
    )
    line = base.mark_line(color="#4da3ff", strokeWidth=2.5)
    points = base.mark_circle(color="#9bd0ff", size=32)
    baseline = alt.Chart(pd.DataFrame({"initial_nav": [initial_nav]})).mark_rule(
        color="#607a96",
        strokeDash=[5, 5],
    ).encode(y="initial_nav:Q")
    return _configure_dark_chart((line + points + baseline).properties(height=240))


def _drawdown_chart(nav_frame: pd.DataFrame) -> alt.Chart:
    chart_frame = nav_frame.copy()
    return _configure_dark_chart(
        alt.Chart(chart_frame)
        .mark_area(color="rgba(220,93,93,0.42)", line={"color": "#dc5d5d"})
        .encode(
            x=alt.X("decision_step:Q", title="Decision Step"),
            y=alt.Y("drawdown:Q", title="Drawdown", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("week_number:Q", title="Week"),
                alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%"),
            ],
        )
        .properties(height=210)
    )


def _allocation_chart(state: PortfolioState) -> alt.Chart:
    market_values = state.market_value_dict()
    allocation_rows = [
        {
            "segment": "Cash",
            "value": float(state.cash),
            "weight": float(state.cash / state.total_nav) if state.total_nav > _EPSILON else 0.0,
            "type": "Cash",
        }
    ]
    for ticker, value in sorted(market_values.items()):
        allocation_rows.append(
            {
                "segment": ticker,
                "value": float(value),
                "weight": float(value / state.total_nav) if state.total_nav > _EPSILON else 0.0,
                "type": "Holding",
            }
        )
    allocation_frame = pd.DataFrame(allocation_rows)
    return _configure_dark_chart(
        alt.Chart(allocation_frame)
        .mark_bar(cornerRadiusEnd=8)
        .encode(
            y=alt.Y("segment:N", title=None, sort="-x"),
            x=alt.X("weight:Q", title="Share of Current Portfolio", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=["Cash", "Holding"], range=["#607a96", "#4da3ff"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("value:Q", title="Value", format=",.2f"),
                alt.Tooltip("weight:Q", title="Weight", format=".2%"),
            ],
        )
        .properties(height=max(150, 40 * len(allocation_rows)))
    )


def _portfolio_insight_chips(state: PortfolioState) -> list[tuple[str, str]]:
    shares = state.shares_dict()
    market_values = state.market_value_dict()
    holdings_count = sum(1 for quantity in shares.values() if quantity > _EPSILON)
    if holdings_count == 0:
        return [("No holdings yet", "neutral"), ("All capital is currently in cash", "good")]

    chips: list[tuple[str, str]] = []
    cash_weight = float(state.cash / state.total_nav) if state.total_nav > _EPSILON else 0.0
    max_weight = (
        max(float(value) / state.total_nav for value in market_values.values())
        if market_values and state.total_nav > _EPSILON
        else 0.0
    )

    if cash_weight > 0.65:
        chips.append(("Most capital is still in cash", "neutral"))
    elif cash_weight < 0.20:
        chips.append(("Most capital is currently invested", "good"))

    if max_weight > 0.50:
        chips.append(("Portfolio is concentrated in one position", "risk"))
    elif holdings_count >= 3 and max_weight < 0.35:
        chips.append(("Portfolio is currently diversified", "good"))

    if len(state.nav_history) >= 2:
        if float(state.total_nav) >= max(state.nav_history) * 0.99:
            chips.append(("Portfolio value is near its visible high", "good"))
        else:
            chips.append(("Portfolio remains below its visible peak", "warn"))
    return chips


def _portfolio_context_summary(state: PortfolioState) -> str:
    holdings_count = sum(1 for quantity in state.shares_dict().values() if quantity > _EPSILON)
    if holdings_count == 0:
        return "You are entirely in cash at the moment, so the portfolio has no active stock exposure."
    max_weight = (
        max(float(value) / state.total_nav for value in state.market_value_dict().values())
        if state.market_value_dict() and state.total_nav > _EPSILON
        else 0.0
    )
    if max_weight > 0.50:
        return "One position currently dominates the portfolio, so concentration risk is elevated."
    if holdings_count >= 3:
        return "Capital is spread across multiple holdings, so the portfolio is less concentrated."
    return "The portfolio currently holds a small number of positions with moderate concentration."


def _signed_currency_text(value: float) -> str:
    if value > _EPSILON:
        return f"+{_currency(abs(value))}"
    if value < -_EPSILON:
        return f"-{_currency(abs(value))}"
    return _currency(0.0)


def _signed_number_text(value: int) -> str:
    if value > 0:
        return f"+{value}"
    return str(value)


def _default_chart_ticker(observation: Observation) -> str:
    held_tickers = sorted(observation.portfolio_state.shares_dict().keys())
    if held_tickers:
        return held_tickers[0]
    return observation.available_tickers[0]


def _watchlist_change_badge_html(change_pct: float | None) -> str:
    """Render a compact change badge for the watchlist."""
    if change_pct is None:
        return (
            "<div style=\""
            "display:inline-block;"
            "padding:0.28rem 0.55rem;"
            "border-radius:999px;"
            "background:rgba(96,122,150,0.16);"
            "border:1px solid rgba(96,122,150,0.22);"
            "color:#b9cad9;"
            "font-weight:700;"
            "font-size:0.82rem;"
            "line-height:1.15;"
            "\">--</div>"
        )

    if change_pct > _EPSILON:
        arrow = "↑"
        text_color = "#72e0a5"
        background_color = "rgba(31,143,95,0.16)"
        border_color = "rgba(31,143,95,0.24)"
    elif change_pct < -_EPSILON:
        arrow = "↓"
        text_color = "#ff8e8e"
        background_color = "rgba(220,93,93,0.16)"
        border_color = "rgba(220,93,93,0.24)"
    else:
        arrow = "→"
        text_color = "#bfd2e2"
        background_color = "rgba(96,122,150,0.16)"
        border_color = "rgba(96,122,150,0.22)"

    return (
        "<div style=\""
        "display:inline-block;"
        "padding:0.28rem 0.55rem;"
        "border-radius:999px;"
        f"background:{background_color};"
        f"border:1px solid {border_color};"
        f"color:{text_color};"
        "font-weight:700;"
        "font-size:0.82rem;"
        "line-height:1.15;"
        "\">"
        f"{arrow} {abs(change_pct):.2%}"
        "</div>"
    )


def _action_summary(action: Action | None) -> str:
    if action is None:
        return "No action"
    if action.action_type == ActionType.HOLD:
        return "Do nothing this week"
    if action.action_type == ActionType.BUY:
        assert action.ticker is not None
        assert action.quantity_type is not None
        assert action.quantity is not None
        if action.quantity_type == QuantityType.SHARES:
            return f"Buy {_format_shares(action.quantity)} of {action.ticker}"
        if action.quantity_type == QuantityType.NOTIONAL_DOLLARS:
            return f"Buy {_currency(action.quantity)} of {action.ticker}"
        return f"Buy {action.quantity:.2%} of portfolio value in {action.ticker}"
    if action.action_type == ActionType.SELL:
        assert action.ticker is not None
        assert action.quantity_type is not None
        if action.quantity_type == QuantityType.CLOSE_ALL:
            return f"Sell all shares of {action.ticker}"
        assert action.quantity is not None
        if action.quantity_type == QuantityType.SHARES:
            return f"Sell {_format_shares(action.quantity)} of {action.ticker}"
        return f"Sell {_currency(action.quantity)} of {action.ticker}"
    if action.action_type == ActionType.REDUCE:
        assert action.ticker is not None
        assert action.fraction is not None
        return f"Reduce {action.ticker} by {action.fraction:.0%}"
    if action.action_type == ActionType.SET_STOP:
        assert action.ticker is not None
        assert action.stop_price is not None
        return f"Set a stop price on {action.ticker} at {_currency(action.stop_price)}"
    if action.action_type == ActionType.REMOVE_STOP:
        assert action.ticker is not None
        return f"Remove the stop price from {action.ticker}"
    return action.action_type.value.replace("_", " ").title()


def _action_detail(action: Action) -> str | None:
    if action.action_type == ActionType.BUY and action.quantity_type == QuantityType.NAV_FRACTION:
        return "This buy is sized as a share of your portfolio value at the start of the week."
    if action.action_type == ActionType.SELL and action.quantity_type == QuantityType.CLOSE_ALL:
        return "The simulator will try to fully close this holding."
    if action.action_type == ActionType.SET_STOP:
        return "If a later weekly low breaches this price, the simulator can schedule a forced sale."
    return None


def _render_feedback_card(title: str, items: Sequence[str], *, tone: str) -> None:
    tone_class = {
        "success": "success",
        "warning": "warning",
        "error": "error",
        "neutral": "neutral",
    }.get(tone, "neutral")
    items_html = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    st.markdown(
        (
            f"<div class='feedback-card feedback-card--{tone_class}'>"
            f"<div class='feedback-title'>{html.escape(title)}</div>"
            f"<ul>{items_html}</ul>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _termination_reason_message(reason: str) -> str:
    if reason == "end_of_data":
        return "The episode has ended because there is no later market week available."
    if reason == "blow_up":
        return "The episode has ended because a configured blow-up threshold was triggered."
    return f"Episode finished because: {reason}"


def _humanize_reason(reason: str) -> str:
    normalized = reason.strip()
    replacements = {
        "Clipped to available cash after estimated execution costs": (
            "The requested size was reduced to fit the cash available after estimated trading costs."
        ),
        "Clipped to remaining weekly turnover budget": (
            "The requested size was reduced to fit the remaining trading budget for this week."
        ),
        "Trade size reduced to zero by hard constraints": (
            "The requested size became too small after the simulator applied its hard rules."
        ),
        "Projected total NAV would become non-positive": (
            "This action would have pushed the portfolio value to zero or below."
        ),
    }
    return replacements.get(normalized, normalized)


def _currency(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _price_change_badge_html(change: float) -> str:
    if change > _EPSILON:
        arrow = "↑"
        text_color = "#72e0a5"
        background_color = "rgba(31,143,95,0.16)"
        border_color = "rgba(31,143,95,0.24)"
    elif change < -_EPSILON:
        arrow = "↓"
        text_color = "#ff8e8e"
        background_color = "rgba(220,93,93,0.16)"
        border_color = "rgba(220,93,93,0.24)"
    else:
        arrow = "→"
        text_color = "#bfd2e2"
        background_color = "rgba(96,122,150,0.16)"
        border_color = "rgba(96,122,150,0.22)"

    value_text = f"{arrow} {_currency(abs(change))}"
    return (
        "<div style=\""
        "display:inline-block;"
        "padding:0.25rem 0.6rem;"
        "border-radius:999px;"
        f"background:{background_color};"
        f"border:1px solid {border_color};"
        f"color:{text_color};"
        "font-weight:700;"
        "font-size:0.92rem;"
        "line-height:1.2;"
        "\">"
        f"{value_text}"
        "</div>"
    )


def _format_shares(value: float) -> str:
    rounded = round(float(value), 4)
    if rounded.is_integer():
        whole_shares = int(rounded)
        unit = "share" if whole_shares == 1 else "shares"
        return f"{whole_shares:,} {unit}"
    return f"{rounded:,.4f} shares"
