from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    CAPTURE_RATE,
    HOSTING_COST,
    OUTPUTS_DIR,
    SPEND_BASE,
    VALUE_SHARE,
)
from src.genai_templates import (
    chat_assistant_response,
    email_to_ap_processor,
    email_to_vendor_manager,
    escalation_note,
)
from src.utils import money, value_calculation

st.set_page_config(
    page_title="Invoice Payment Governance Agent",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
:root {
    --bg1: #07111f;
    --bg2: #0e1a2f;
    --card: rgba(15, 23, 42, 0.72);
    --card-2: rgba(30, 41, 59, 0.78);
    --line: rgba(255,255,255,0.08);
    --text: #f8fafc;
    --muted: #94a3b8;
    --pink: #f72585;
    --violet: #8b5cf6;
    --cyan: #22d3ee;
    --green: #22c55e;
    --yellow: #f59e0b;
    --red: #ef4444;
}
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: radial-gradient(circle at top left, #152648 0%, var(--bg1) 32%, #050b16 100%);
    color: var(--text);
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"] { right: 1rem; }
#MainMenu, footer {visibility: hidden;}
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
    max-width: 98vw;
}
h1, h2, h3, h4, h5, h6, p, label, span, div { color: var(--text); }
section[data-testid="stSidebar"] { display:none; }
.hero {
    background: linear-gradient(135deg, rgba(139,92,246,.24), rgba(34,211,238,.14), rgba(247,37,133,.18));
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 16px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,.25);
    backdrop-filter: blur(12px);
    margin-bottom: 10px;
}
.hero-title {
    font-size: 1.85rem; font-weight: 900; margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.hero-subtitle { color: var(--muted); font-size: .95rem; }
.metric-card {
    background: linear-gradient(180deg, rgba(17,24,39,.88), rgba(15,23,42,.78));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    padding: 14px 16px;
    min-height: 98px;
    box-shadow: 0 8px 28px rgba(0,0,0,.18);
}
.metric-label {
    color: #cbd5e1;
    font-size: .74rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 700;
    margin-bottom: 6px;
}
.metric-value {
    color: white;
    font-size: 1.38rem;
    line-height: 1.1;
    font-weight: 900;
}
.metric-sub { color: var(--muted); font-size: .78rem; margin-top: 6px; }
.metric-line {
    height: 4px; width: 100%; border-radius: 999px; margin-top: 12px;
    background: linear-gradient(90deg, var(--cyan), var(--violet), var(--pink));
    opacity: .85;
}
.panel {
    background: linear-gradient(180deg, rgba(15,23,42,.85), rgba(15,23,42,.70));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    padding: 12px 14px 10px 14px;
    box-shadow: 0 8px 28px rgba(0,0,0,.16);
    margin-bottom: 8px;
}
.panel-title {
    font-size: 1rem; font-weight: 800; margin-bottom: 6px;
}
.small-note { color: var(--muted); font-size: .8rem; }
.badge {
    padding: 5px 11px; border-radius: 999px; font-weight: 800; font-size: .76rem; display: inline-block;
    border: 1px solid rgba(255,255,255,.06);
}
.badge-red {background: rgba(239,68,68,.18); color: #fecaca;}
.badge-yellow {background: rgba(245,158,11,.18); color: #fde68a;}
.badge-green {background: rgba(34,197,94,.18); color: #bbf7d0;}
.badge-blue {background: rgba(34,211,238,.18); color: #a5f3fc;}
.badge-purple {background: rgba(139,92,246,.18); color: #ddd6fe;}
div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15,23,42,.40);
    padding: 4px;
    border-radius: 18px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 14px;
    height: 42px;
    padding: 0 16px;
    color: #e2e8f0;
    background: rgba(255,255,255,.03);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(139,92,246,.80), rgba(34,211,238,.72)) !important;
    color: white !important;
}
.stRadio > div { gap: 0.35rem; }
.stRadio [role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    background: rgba(15,23,42,.55);
    padding: 7px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,.07);
}
.stRadio [role="radiogroup"] label {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 14px;
    padding: 7px 12px;
    margin: 0;
}
.stMultiSelect, .stSelectbox, .stNumberInput, .stTextInput, .stSlider {
    background: transparent;
}
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
    background: rgba(15,23,42,.72) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 14px !important;
    color: white !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(17,24,39,.88), rgba(15,23,42,.78));
    border: 1px solid rgba(255,255,255,.07);
    padding: 10px 14px;
    border-radius: 18px;
}
[data-testid="stMetricLabel"] { color: #cbd5e1; }
[data-testid="stMetricValue"] { color: white; }
button[kind="secondary"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,.08) !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
    font=dict(color="#E2E8F0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
COLOR_SEQ = ["#22d3ee", "#8b5cf6", "#f72585", "#22c55e", "#f59e0b", "#60a5fa"]


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_data():
    scored = load_csv(OUTPUTS_DIR / "scored_invoices.csv")
    flagged = load_csv(OUTPUTS_DIR / "flagged_invoices.csv")
    audit = load_csv(OUTPUTS_DIR / "audit_trail.csv")
    follow = load_csv(OUTPUTS_DIR / "follow_up_tracker.csv")
    metrics = load_csv(OUTPUTS_DIR / "model_metrics.csv")
    root_metrics = load_csv(OUTPUTS_DIR / "root_cause_metrics.csv")
    due_metrics = load_csv(OUTPUTS_DIR / "due_date_metrics.csv")
    ocr_metrics = load_csv(OUTPUTS_DIR / "ocr_metrics.csv")
    feature_importance = load_csv(OUTPUTS_DIR / "feature_importance.csv")
    forecast = load_csv(OUTPUTS_DIR / "leakage_forecast.csv")
    validation = load_csv(OUTPUTS_DIR / "data_validation_report.csv")
    return scored, flagged, audit, follow, metrics, root_metrics, due_metrics, ocr_metrics, feature_importance, forecast, validation


def kpi_card(label: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-sub'>{subtitle}</div>
            <div class='metric-line'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_title(title: str, note: str = ""):
    st.markdown(
        f"<div class='panel'><div class='panel-title'>{title}</div><div class='small-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def status_badge(text: str):
    cls = "badge-blue"
    if text in ["HOLD", "ESCALATE", "Review Required", "Escalated", "Critical"]:
        cls = "badge-red"
    elif text in ["DEFER", "Deferred", "High"]:
        cls = "badge-yellow"
    elif text in ["APPROVE", "Ready to Release", "Low"]:
        cls = "badge-green"
    elif text in ["Insufficient audit trail", "Behavioural", "System error", "UPR", "Not applicable / on-time"]:
        cls = "badge-purple"
    st.markdown(f"<span class='badge {cls}'>{text}</span>", unsafe_allow_html=True)


def smart_dataframe(df: pd.DataFrame, height: int = 280):
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def ensure_pipeline_hint(scored: pd.DataFrame):
    if scored.empty:
        st.warning("No generated outputs found. Run the pipeline first: `python run_pipeline.py`, then refresh the app.")
        if st.button("Run pipeline now", key="run_pipeline_button_main"):
            with st.spinner("Running pipeline. This may take a few moments..."):
                result = subprocess.run([sys.executable, "run_pipeline.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Pipeline completed. Refresh the page to load outputs.")
                    st.code(result.stdout[-2500:])
                else:
                    st.error("Pipeline failed.")
                    st.code(result.stderr[-4000:])
        st.stop()


scored, flagged, audit, follow, metrics, root_metrics, due_metrics, ocr_metrics, feature_importance, forecast, validation = load_data()
ensure_pipeline_hint(scored)

for c in ["invoice_amount", "risk_score", "days_paid_early"]:
    if c in scored.columns:
        scored[c] = pd.to_numeric(scored[c], errors="coerce")
for date_col in ["contractual_due_date", "sap_payment_release_date", "recommended_release_date", "sap_due_date"]:
    if date_col in scored.columns:
        scored[date_col] = pd.to_datetime(scored[date_col], errors="coerce")

if "spend_base" not in st.session_state:
    st.session_state.spend_base = float(SPEND_BASE)
    st.session_state.early_payment_rate = 0.02
    st.session_state.capture_rate = float(CAPTURE_RATE)
    st.session_state.value_share = float(VALUE_SHARE)
    st.session_state.hosting_cost = float(HOSTING_COST)
    st.session_state.risk_threshold = 0.60

value = value_calculation(
    st.session_state.spend_base,
    st.session_state.early_payment_rate,
    st.session_state.capture_rate,
    st.session_state.value_share,
    st.session_state.hosting_cost,
)

early = scored[scored["early_payment_flag"] == 1].copy() if "early_payment_flag" in scored.columns else pd.DataFrame()

st.markdown(
    """
    <div class='hero'>
        <div class='hero-title'>AI-Powered Invoice Payment Governance Agent</div>
        <div class='hero-subtitle'>Interactive governance cockpit for early payment risk, root cause analytics, intervention workflow, and working-capital leakage prevention.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Executive Overview",
    "Risk Queue",
    "Invoice Detail View",
    "Root Cause Analytics",
    "Vendor and Processor Trends",
    "Model Performance",
    "Explainability",
    "Forecasting",
    "Follow-up Tracker",
    "Audit Trail",
    "Configuration",
]
page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed", key="top_nav_page")

if page == "Executive Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Invoices", f"{len(scored):,}", "Invoices in monitoring scope")
    with c2:
        kpi_card("Invoice Value", money(scored["invoice_amount"].sum()), "Total payment value")
    with c3:
        kpi_card("Early Payments", f"{len(early):,}", "Detected or simulated early cases")
    with c4:
        kpi_card("Early Value", money(early["invoice_amount"].sum() if not early.empty else 0), "Potential leakage value")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi_card("WC At Risk", money(value["early_payment_exposure"]), "Spend base × early payment rate")
    with c6:
        kpi_card("Retained WC", money(value["retained_working_capital"]), "Captured via intervention")
    with c7:
        kpi_card("Value Share Revenue", money(value["genpact_value_share_revenue"]), "Business value realization")
    with c8:
        kpi_card("ROI Multiple", f"{value['roi_multiple']:.1f}x", "Revenue ÷ hosting cost")

    col1, col2 = st.columns([1.25, 0.75])
    with col1:
        panel_title("Leakage Trend", "Historical and forecasted early-payment leakage")
        if not forecast.empty:
            forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
            fig = px.line(
                forecast,
                x="ds",
                y="early_payment_value",
                color="forecast_type",
                markers=True,
                color_discrete_sequence=COLOR_SEQ,
                height=300,
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True, key="exec_forecast_plot")
        else:
            st.info("Forecast output not found.")
    with col2:
        panel_title("Root Cause Mix", "Distribution of predicted early-payment root causes")
        rc = scored["predicted_root_cause"].fillna("Insufficient audit trail").value_counts().reset_index()
        rc.columns = ["root_cause", "count"]
        fig2 = px.pie(rc, names="root_cause", values="count", hole=0.62, color_discrete_sequence=COLOR_SEQ, height=300)
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True, key="exec_root_cause_pie")

    with st.expander("Leadership Insight + Scenario View", expanded=True):
        c9, c10 = st.columns([1.1, 0.9])
        with c9:
            st.info(
                f"The POC identified {len(early):,} early-payment cases worth {money(early['invoice_amount'].sum() if not early.empty else 0)}. "
                f"Using current assumptions, the retained working-capital opportunity is {money(value['retained_working_capital'])}."
            )
        with c10:
            if "scenario_label" in scored.columns:
                scen = scored["scenario_label"].value_counts().reset_index()
                scen.columns = ["scenario", "count"]
                smart_dataframe(scen, height=220)

elif page == "Risk Queue":
    header_col, filter_col = st.columns([0.72, 0.28])
    with header_col:
        panel_title("Risk Queue", "Compact review queue with internal table scrolling to keep the screen clean")
    with filter_col:
        with st.popover("⚙️ Filters", use_container_width=True):
            market = st.multiselect("Market", sorted(scored["market"].dropna().unique()), key="risk_market_filter")
            vendor = st.multiselect("Vendor", sorted(scored["vendor_name"].dropna().unique())[:200], key="risk_vendor_filter")
            root = st.multiselect("Root cause", sorted(scored["predicted_root_cause"].dropna().unique()), key="risk_root_filter")
            action = st.multiselect("Action", sorted(scored["recommended_action"].dropna().unique()), key="risk_action_filter")
            threshold = st.slider("Risk threshold", 0.0, 1.0, st.session_state.risk_threshold, 0.05, key="risk_threshold_slider_queue")
    q = scored.copy()
    market = st.session_state.get("risk_market_filter", [])
    vendor = st.session_state.get("risk_vendor_filter", [])
    root = st.session_state.get("risk_root_filter", [])
    action = st.session_state.get("risk_action_filter", [])
    threshold = st.session_state.get("risk_threshold_slider_queue", st.session_state.risk_threshold)
    if market:
        q = q[q["market"].isin(market)]
    if vendor:
        q = q[q["vendor_name"].isin(vendor)]
    if root:
        q = q[q["predicted_root_cause"].isin(root)]
    if action:
        q = q[q["recommended_action"].isin(action)]
    q = q[q["risk_score"] >= threshold]
    cols = [
        "invoice_id", "vendor_name", "market", "currency", "invoice_amount", "contractual_due_date",
        "sap_payment_release_date", "days_paid_early", "risk_score", "predicted_root_cause", "recommended_action", "status"
    ]
    if "scenario_label" in q.columns:
        cols.insert(1, "scenario_label")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Rows", f"{len(q):,}")
    with a2:
        st.metric("Avg Risk", f"{q['risk_score'].mean():.2f}" if not q.empty else "0.00")
    with a3:
        st.metric("High Risk", f"{(q['risk_score'] >= 0.8).sum():,}" if not q.empty else "0")
    with a4:
        st.download_button("Download CSV", q[cols].to_csv(index=False), "risk_queue.csv", "text/csv", key="risk_queue_download")
    if q.empty:
        st.info("No rows found for selected filters.")
    else:
        smart_dataframe(q[cols].sort_values("risk_score", ascending=False), height=420)

elif page == "Invoice Detail View":
    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        selected = st.selectbox(
            "Select invoice",
            scored.sort_values("risk_score", ascending=False)["invoice_id"].tolist(),
            key="invoice_detail_selectbox",
        )
    row = scored[scored["invoice_id"] == selected].iloc[0]
    with c2:
        info1, info2, info3, info4 = st.columns(4)
        with info1:
            kpi_card("Risk Score", f"{row['risk_score']:.2f}")
        with info2:
            kpi_card("Days Early", f"{int(row['days_paid_early'])}")
        with info3:
            kpi_card("Amount", f"{row['currency']} {row['invoice_amount']:,.0f}")
        with info4:
            st.markdown("<div style='padding-top:12px'></div>", unsafe_allow_html=True)
            status_badge(str(row["recommended_action"]))
    t1, t2, t3, t4 = st.tabs(["Invoice Snapshot", "Due-Date Governance", "Explanation + Email", "Audit"])
    with t1:
        snap1, snap2 = st.columns([0.48, 0.52])
        with snap1:
            fields = [
                "invoice_id", "vendor_name", "market", "currency", "invoice_amount", "invoice_priority",
                "approval_status", "processor_name", "predicted_root_cause", "scenario_label"
            ]
            fields = [f for f in fields if f in row.index]
            smart_dataframe(pd.DataFrame({"field": fields, "value": [row[f] for f in fields]}), height=300)
        with snap2:
            st.write("**Rule Flags**")
            st.warning(row.get("rule_flags", "No rule flags"))
            st.write("**Root Cause**")
            status_badge(str(row.get("predicted_root_cause", "Insufficient audit trail")))
            st.write("<br>", unsafe_allow_html=True)
            st.write(row.get("root_cause_narrative", "No root cause narrative available"))
    with t2:
        due_df = pd.DataFrame([
            {"Source": "Contractual Due Date", "Date": row.get("contractual_due_date")},
            {"Source": "SAP Due Date", "Date": row.get("sap_due_date")},
            {"Source": "Scheduled Release Date", "Date": row.get("sap_payment_release_date")},
            {"Source": "Recommended Release Date", "Date": row.get("recommended_release_date")},
        ])
        smart_dataframe(due_df, height=260)
    with t3:
        left, right = st.columns([0.46, 0.54])
        with left:
            st.write("**Plain-Language Explanation**")
            st.info(row.get("risk_explanation", "No explanation available"))
            question = st.text_input("Ask invoice assistant", "Why was this invoice flagged?", key="invoice_chat_question")
            st.write(chat_assistant_response(question, row))
        with right:
            email_tab1, email_tab2, email_tab3 = st.tabs(["AP Processor", "Vendor Manager", "Finance Escalation"])
            with email_tab1:
                st.code(email_to_ap_processor(row), language="text")
            with email_tab2:
                st.code(email_to_vendor_manager(row), language="text")
            with email_tab3:
                st.code(escalation_note(row), language="text")
    with t4:
        inv_audit = audit[audit["invoice_id"] == selected] if not audit.empty else pd.DataFrame()
        if inv_audit.empty:
            st.info("No audit records for this invoice.")
        else:
            smart_dataframe(inv_audit, height=300)

elif page == "Root Cause Analytics":
    a, b, c = st.columns([0.34, 0.33, 0.33])
    rc = scored["predicted_root_cause"].fillna("Insufficient audit trail").value_counts().reset_index()
    rc.columns = ["root_cause", "count"]
    by_market = scored.groupby(["market", "predicted_root_cause"]).size().reset_index(name="count")
    by_proc = (
        scored.groupby(["processor_name", "predicted_root_cause"]).size().reset_index(name="count")
        .sort_values("count", ascending=False).head(20)
    )
    with a:
        fig = px.bar(rc, x="root_cause", y="count", color="root_cause", color_discrete_sequence=COLOR_SEQ, height=300)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="rc_dist_bar")
    with b:
        fig = px.bar(by_market, x="market", y="count", color="predicted_root_cause", color_discrete_sequence=COLOR_SEQ, height=300)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, key="rc_market_bar")
    with c:
        fig = px.bar(by_proc, x="processor_name", y="count", color="predicted_root_cause", color_discrete_sequence=COLOR_SEQ, height=300)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, key="rc_proc_bar")
    audit_gap = scored[(scored["predicted_root_cause"] == "Insufficient audit trail") & (scored["risk_score"] >= 0.60)]
    with st.expander("Insufficient Audit Trail Priority Queue", expanded=True):
        cols = ["invoice_id", "vendor_name", "market", "invoice_amount", "risk_score", "recommended_action", "rule_flags"]
        smart_dataframe(audit_gap[cols].head(300), height=260)

elif page == "Vendor and Processor Trends":
    early_df = scored[scored["early_payment_flag"] == 1].copy()
    v = (
        early_df.groupby("vendor_name")
        .agg(early_payment_count=("invoice_id", "count"), early_payment_value=("invoice_amount", "sum"), avg_risk=("risk_score", "mean"))
        .reset_index().sort_values("early_payment_value", ascending=False).head(15)
    )
    p = (
        early_df.groupby("processor_name")
        .agg(early_payment_count=("invoice_id", "count"), avg_manual_override=("processor_manual_override_rate", "mean"), avg_risk=("risk_score", "mean"))
        .reset_index().sort_values("early_payment_count", ascending=False).head(15)
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(v, x="vendor_name", y="early_payment_value", color="avg_risk", color_continuous_scale="tealrose", height=310)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, key="vendor_value_bar")
    with c2:
        fig = px.bar(p, x="processor_name", y="early_payment_count", color="avg_manual_override", color_continuous_scale="sunset", height=310)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, key="processor_count_bar")
    e1, e2 = st.columns(2)
    with e1:
        with st.expander("Repeat Offender Vendors", expanded=True):
            smart_dataframe(v, height=240)
    with e2:
        with st.expander("Processor Override Trend", expanded=True):
            smart_dataframe(p, height=240)

elif page == "Model Performance":
    benchmark = load_csv(OUTPUTS_DIR / "benchmark_metrics.csv")
    scenario_metrics = load_csv(OUTPUTS_DIR / "scenario_m1_metrics.csv")
    root_cm = load_csv(OUTPUTS_DIR / "root_cause_confusion_matrix.csv")

    if not metrics.empty and "f1" in metrics.columns:
        valid = metrics.dropna(subset=["f1"]).sort_values("f1", ascending=False)
        best_row = valid.iloc[0] if not valid.empty else None
    else:
        best_row = None

    if not benchmark.empty:
        b = benchmark.set_index("model")
        ml_rows = [idx for idx in b.index if str(idx).startswith("Best ML Model")]
        legacy_row = b.loc["Legacy Rule Guardrail"] if "Legacy Rule Guardrail" in b.index else None
        ml_row = b.loc[ml_rows[0]] if ml_rows else None
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Legacy F1", f"{legacy_row['f1']:.2f}" if legacy_row is not None else "NA", "Rule-only guardrail")
        with c2:
            kpi_card("Best ML F1", f"{ml_row['f1']:.2f}" if ml_row is not None else "NA", "Model-based risk scorer")
        with c3:
            uplift = (ml_row['f1'] - legacy_row['f1']) if legacy_row is not None and ml_row is not None else 0
            kpi_card("F1 Uplift", f"{uplift:+.2f}", "ML vs rule baseline")
        with c4:
            kpi_card("Target F1", "0.77", "POC success benchmark")
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            kpi_card("Best M1 Model", best_row["model"] if best_row is not None else "NA", "Selected on F1 + recall")
        with m2:
            kpi_card("Best F1", f"{best_row['f1']:.2f}" if best_row is not None else "NA")
        with m3:
            kpi_card("Best Recall", f"{best_row['recall']:.2f}" if best_row is not None else "NA")
        with m4:
            kpi_card("Best ROC-AUC", f"{best_row['roc_auc']:.2f}" if best_row is not None else "NA")

    tab1, tab2, tab3, tab4 = st.tabs(["M1 Risk Scorer", "Scenario Performance", "M2/M3/M4", "Validation"])
    with tab1:
        st.caption("M1 excludes direct target-leakage fields such as days_paid_early and days_to_contractual_due_date. Benchmark cards compare the ML model against a legacy rule-only guardrail.")
        left, right = st.columns([0.52, 0.48])
        with left:
            smart_dataframe(metrics, height=230)
            if not benchmark.empty:
                st.write("**Benchmark vs Baseline**")
                smart_dataframe(benchmark, height=170)
        with right:
            cm = load_csv(OUTPUTS_DIR / "early_payment_confusion_matrix.csv")
            if not cm.empty:
                try:
                    cm.columns = [str(c) for c in cm.columns]
                    cm.index = [str(i) for i in cm.index]
                    desired_order = [lbl for lbl in ["0", "1"] if lbl in cm.index and lbl in cm.columns]
                    if len(desired_order) == 2:
                        cm = cm.loc[desired_order, desired_order]
                except Exception:
                    pass
                zvals = cm.values
                heat = go.Figure(
                    data=go.Heatmap(
                        z=zvals,
                        x=[f"Pred {c}" for c in cm.columns.astype(str)],
                        y=[f"Actual {i}" for i in cm.index.astype(str)],
                        text=zvals,
                        texttemplate="%{text}",
                        textfont={"size": 18, "color": "white"},
                        colorscale=[[0.0, "#172554"], [0.35, "#0891b2"], [0.7, "#7c3aed"], [1.0, "#22c55e"]],
                        hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
                    )
                )
                heat.update_layout(**PLOTLY_LAYOUT, height=300, title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="Actual Label")
                heat.update_xaxes(side="top", showgrid=False)
                heat.update_yaxes(autorange="reversed", showgrid=False)
                st.plotly_chart(heat, use_container_width=True, key="m1_cm_heatmap")
        if not feature_importance.empty:
            fig = px.bar(feature_importance.head(15), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Turbo", height=310)
            fig.update_layout(**PLOTLY_LAYOUT, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True, key="feature_importance_plot")

    with tab2:
        if scenario_metrics.empty:
            st.info("Scenario-wise performance is generated after running python run_pipeline.py.")
        else:
            left, right = st.columns([0.58, 0.42])
            with left:
                smart_dataframe(scenario_metrics.sort_values("support", ascending=False), height=340)
            with right:
                plot_df = scenario_metrics.sort_values("f1", ascending=True)
                fig = px.bar(plot_df, x="f1", y="scenario_label", orientation="h", color="positive_rate", color_continuous_scale="Tealrose", height=340)
                fig.update_layout(**PLOTLY_LAYOUT, title="M1 F1 by Business Scenario")
                st.plotly_chart(fig, use_container_width=True, key="scenario_f1_chart")

    with tab3:
        x1, x2, x3 = st.columns([0.34, 0.33, 0.33])
        with x1:
            st.write("**M2 Root Cause Classifier**")
            smart_dataframe(root_metrics, height=220)
        with x2:
            st.write("**M3 OCR Accuracy**")
            smart_dataframe(ocr_metrics, height=220)
        with x3:
            st.write("**M4 Due Date Recommender**")
            smart_dataframe(due_metrics, height=220)
        if not root_cm.empty:
            st.write("**M2 Root Cause Confusion Matrix**")
            first_col = root_cm.columns[0]
            if first_col.lower().startswith("unnamed"):
                root_cm = root_cm.set_index(first_col)
            zvals = root_cm.values
            heat = go.Figure(data=go.Heatmap(
                z=zvals,
                x=[str(c) for c in root_cm.columns],
                y=[str(i) for i in root_cm.index],
                text=zvals,
                texttemplate="%{text}",
                textfont={"size": 13, "color": "white"},
                colorscale=[[0.0, "#111827"], [0.35, "#1d4ed8"], [0.7, "#9333ea"], [1.0, "#10b981"]],
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            heat.update_layout(**PLOTLY_LAYOUT, height=360, title="Root Cause Confusion Matrix", xaxis_title="Predicted Root Cause", yaxis_title="Actual Root Cause")
            heat.update_xaxes(side="top")
            heat.update_yaxes(autorange="reversed")
            st.plotly_chart(heat, use_container_width=True, key="root_cm_heatmap")
        loss = load_csv(OUTPUTS_DIR / "pytorch_loss_curve.csv")
        if not loss.empty:
            fig = px.line(loss, x="epoch", y="loss", markers=True, color_discrete_sequence=["#22d3ee"], height=240)
            fig.update_layout(**PLOTLY_LAYOUT, title="PyTorch Loss Curve")
            st.plotly_chart(fig, use_container_width=True, key="pytorch_loss_plot")

    with tab4:
        smart_dataframe(validation, height=360)

elif page == "Explainability":
    st.caption("Explainability view for invoice-level decisioning: M1 risk factors, M2 root-cause evidence, and M4 due-date adjustment logic.")
    invoice_ids = scored.sort_values("risk_score", ascending=False)["invoice_id"].tolist()
    selected = st.selectbox("Select invoice for explainability", invoice_ids, key="explain_invoice_select")
    row = scored[scored["invoice_id"] == selected].iloc[0]
    due_recs = load_csv(OUTPUTS_DIR / "due_date_recommendations.csv")
    due_effects = load_csv(OUTPUTS_DIR / "due_date_feature_effects.csv")

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        kpi_card("M1 Risk Score", f"{row['risk_score']:.2f}", "Probability of early payment risk")
    with e2:
        kpi_card("M2 Root Cause", str(row.get("predicted_root_cause", "NA")), "Predicted governance category")
    with e3:
        kpi_card("M4 Recommended Date", str(row.get("recommended_release_date", "NA"))[:10], "Release-date recommendation")
    with e4:
        kpi_card("Action", str(row.get("recommended_action", "NA")), "Hold / approve / defer / escalate")

    c1, c2 = st.columns([0.48, 0.52])
    with c1:
        st.write("**M1: Top Global Risk Drivers**")
        if not feature_importance.empty:
            fig = px.bar(feature_importance.head(12), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Turbo", height=310)
            fig.update_layout(**PLOTLY_LAYOUT, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True, key="explain_m1_features")
        st.write("**Invoice Rule Flags**")
        st.warning(row.get("rule_flags", "No rule flags"))
    with c2:
        st.write("**M2: Root-Cause Evidence**")
        evidence = pd.DataFrame([
            {"signal": "Workflow note", "value": row.get("workflow_notes", "")},
            {"signal": "Manual override", "value": row.get("manual_override_flag", "")},
            {"signal": "SAP date delta", "value": row.get("sap_contract_due_date_delta", "")},
            {"signal": "Processor early-release rate", "value": row.get("processor_early_release_rate", "")},
            {"signal": "UPR documented", "value": row.get("upr_documented_flag", "")},
        ])
        smart_dataframe(evidence, height=230)
        st.info(row.get("root_cause_narrative", "No narrative available"))

    d1, d2 = st.columns([0.55, 0.45])
    with d1:
        st.write("**M4: Due-Date Recommendation Logic**")
        if not due_recs.empty and "invoice_id" in due_recs.columns:
            rec = due_recs[due_recs["invoice_id"] == selected]
            show_cols = [c for c in ["invoice_id", "rule_based_recommended_release_date", "true_recommended_release_date", "predicted_adjustment_days", "override_confidence", "recommended_release_date"] if c in due_recs.columns]
            smart_dataframe(rec[show_cols].head(1), height=160)
        else:
            st.info("Due-date recommendation detail not available.")
    with d2:
        st.write("**M4: Top Adjustment Drivers**")
        if not due_effects.empty:
            fig = px.bar(due_effects.head(10), x="abs_coefficient", y="feature", orientation="h", color="coefficient", color_continuous_scale="RdBu", height=260)
            fig.update_layout(**PLOTLY_LAYOUT, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True, key="m4_effects_chart")
        else:
            st.info("M4 feature effects not available.")

elif page == "Forecasting":
    if forecast.empty:
        st.info("Forecast output not available.")
    else:
        forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
        a, b = st.columns(2)
        with a:
            fig = px.line(forecast, x="ds", y="early_payment_value", color="forecast_type", markers=True, color_discrete_sequence=COLOR_SEQ, height=320)
            fig.update_layout(**PLOTLY_LAYOUT, title="Leakage Forecast")
            st.plotly_chart(fig, use_container_width=True, key="forecast_main_plot")
        with b:
            fig = px.line(forecast, x="ds", y="future_retained_with_intervention", color="forecast_type", markers=True, color_discrete_sequence=COLOR_SEQ, height=320)
            fig.update_layout(**PLOTLY_LAYOUT, title="Retained Value with Intervention")
            st.plotly_chart(fig, use_container_width=True, key="forecast_intervention_plot")
        with st.expander("Forecast Data Table"):
            smart_dataframe(forecast, height=220)

elif page == "Follow-up Tracker":
    if follow.empty:
        st.info("No follow-up records found.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Open", f"{(follow['follow_up_status'] == 'Open').sum():,}")
        with c2:
            kpi_card("Escalated", f"{(follow['follow_up_status'] == 'Escalated').sum():,}")
        with c3:
            kpi_card("Pending", f"{(follow['latest_response'] == 'Pending response').sum():,}")
        with c4:
            kpi_card("Closed", f"{(follow['follow_up_status'] == 'Closed').sum():,}")
        x1, x2 = st.columns([0.28, 0.72])
        with x1:
            status = st.multiselect("Status", sorted(follow["follow_up_status"].dropna().unique()), key="follow_status_filter")
        f = follow if not status else follow[follow["follow_up_status"].isin(status)]
        with x2:
            st.download_button("Download Tracker", f.to_csv(index=False), "follow_up_tracker.csv", "text/csv", key="follow_download_button")
        if f.empty:
            st.info("No rows found for selected follow-up filters.")
        else:
            smart_dataframe(f, height=380)

elif page == "Audit Trail":
    if audit.empty:
        st.info("No audit records found.")
    else:
        c1, c2 = st.columns([0.32, 0.68])
        with c1:
            action = st.multiselect("Action", sorted(audit["action_taken"].dropna().unique()), key="audit_action_filter")
        with c2:
            a = audit if not action else audit[audit["action_taken"].isin(action)]
            st.download_button("Download Audit", a.to_csv(index=False), "audit_trail.csv", "text/csv", key="audit_download_button")
        if a.empty:
            st.info("No rows found for selected actions.")
        else:
            smart_dataframe(a, height=420)

elif page == "Configuration":
    left, right = st.columns([0.45, 0.55])
    with left:
        st.session_state.risk_threshold = st.slider("Risk threshold", 0.0, 1.0, st.session_state.risk_threshold, 0.05, key="config_risk_threshold_slider")
        st.session_state.spend_base = st.number_input("Spend base", min_value=0.0, value=st.session_state.spend_base, step=1000000.0, key="config_spend_base_input")
        st.session_state.early_payment_rate = st.number_input("Early payment rate", min_value=0.0, max_value=1.0, value=st.session_state.early_payment_rate, step=0.005, key="config_early_payment_rate_input")
        st.session_state.capture_rate = st.number_input("Capture/interception rate", min_value=0.0, max_value=1.0, value=st.session_state.capture_rate, step=0.01, key="config_capture_rate_input")
        st.session_state.value_share = st.number_input("Value-share percentage", min_value=0.0, max_value=1.0, value=st.session_state.value_share, step=0.01, key="config_value_share_input")
        st.session_state.hosting_cost = st.number_input("Hosting cost", min_value=0.0, value=st.session_state.hosting_cost, step=1000.0, key="config_hosting_cost_input")
    updated = value_calculation(
        st.session_state.spend_base,
        st.session_state.early_payment_rate,
        st.session_state.capture_rate,
        st.session_state.value_share,
        st.session_state.hosting_cost,
    )
    with right:
        c3, c4 = st.columns(2)
        with c3:
            kpi_card("Early Payment Exposure", money(updated["early_payment_exposure"]))
            kpi_card("Retained Working Capital", money(updated["retained_working_capital"]))
        with c4:
            kpi_card("Value Share Revenue", money(updated["genpact_value_share_revenue"]))
            kpi_card("Net Value", money(updated["net_value"]))
        st.success(f"Estimated ROI multiple: {updated['roi_multiple']:.1f}x")
