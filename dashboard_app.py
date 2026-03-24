"""
SPX Mid-Month Strategy v9.2 — Interactive Dashboard
Run: streamlit run dashboard_app.py
"""

import sys, os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(__file__))
from spx_midmonth_strategy import (
    load_data, build_monthly, add_v92_features,
    get_v92_features, get_eom_features,
    walk_forward_v92, walk_forward_eom_v92,
    build_base_position_history,
    build_softrisk_position_history, build_logitstrength_position_history,
    build_backtest_df, build_live_dashboard, summarize_strategy_stats,
    max_drawdown, annualized_sharpe, sigmoid, _safe,
    SOFT_CFGS, SOFT_CFGS_EOM, TEST_START,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPX Mid-Month v9.2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #1e2630;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
}
.block-container { padding-top: 1.5rem; }
div[data-testid="stHorizontalBlock"] { gap: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CACHED PIPELINE
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Running pipeline — midmonth + EOM (~5 min first load)…")
def load_all(oos_start: str):
    df_daily, macro = load_data()
    df_m    = build_monthly(df_daily, macro)
    df_v92  = add_v92_features(df_m)

    # ── Midmonth walk-forward ──
    features    = get_v92_features(df_v92)
    df_wf       = walk_forward_v92(df_v92, features)
    SIGS        = ["LogReg_v92", "ElasticNet_v92", "Ensemble_v92"]
    signal_hist = df_wf[[s for s in SIGS if s in df_wf.columns]].copy()
    base_hist   = build_base_position_history(signal_hist, base_scale=1.5)
    soft_hist   = build_softrisk_position_history(signal_hist, df_v92, SOFT_CFGS)
    ls_hist     = build_logitstrength_position_history(signal_hist)
    all_hist    = pd.concat([base_hist, soft_hist, ls_hist], axis=1)
    all_hist    = all_hist.loc[:, ~all_hist.columns.duplicated(keep="first")]

    # ── EOM walk-forward ──
    eom_features = get_eom_features(df_v92)
    df_wf_eom    = walk_forward_eom_v92(df_v92, eom_features)
    EOM_SIGS     = ["LogReg_v92_EOM", "ElasticNet_v92_EOM", "Ensemble_v92_EOM"]
    sig_hist_eom = df_wf_eom[[s for s in EOM_SIGS if s in df_wf_eom.columns]].copy()
    base_eom     = build_base_position_history(sig_hist_eom, base_scale=1.5)
    soft_eom     = build_softrisk_position_history(sig_hist_eom, df_v92, SOFT_CFGS_EOM)
    ls_eom       = build_logitstrength_position_history(sig_hist_eom)
    eom_hist     = pd.concat([base_eom, soft_eom, ls_eom], axis=1)
    eom_hist     = eom_hist.loc[:, ~eom_hist.columns.duplicated(keep="first")]

    # ── Backtest + live dashboard ──
    backtest_df = build_backtest_df(all_hist, df_v92,
                                    eom_hist=eom_hist, oos_start=oos_start)
    current_df, _ = build_live_dashboard(df_wf, df_v92, SOFT_CFGS,
                                          oos_start=oos_start,
                                          df_wf_eom=df_wf_eom,
                                          soft_cfg_eom_dict=SOFT_CFGS_EOM)

    return dict(
        backtest_df = backtest_df,
        current_df  = current_df,
        df_wf       = df_wf,
        df_wf_eom   = df_wf_eom,
        df_v92      = df_v92,
        all_hist    = all_hist,
        eom_hist    = eom_hist,
    )

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 SPX Mid-Month v9.2")
    st.divider()

    st.subheader("Execution Family")
    sel_families = st.multiselect(
        "Strategy type",
        ["Base", "SoftRisk", "LogitStrength"],
        default=["SoftRisk"],
        help="Base=softsign, SoftRisk=risk-shrinkage, LogitStrength=pct-rank sizing",
    )

    st.subheader("Direction")
    sel_variants = st.multiselect(
        "Long-Only / Long-Short",
        ["LO", "LS"],
        default=["LO", "LS"],
    )

    st.subheader("Signal Source")
    sel_sources = st.multiselect(
        "Entry timing",
        ["Midmonth", "EOM", "Combined"],
        default=["Midmonth", "Combined"],
        help=(
            "Midmonth = signal at day~10 → earns remaining_ret\n"
            "EOM = signal at month-end → earns next-month som_ret\n"
            "Combined = both signals together, full-month P&L"
        ),
    )

    st.subheader("Signal Model")
    sel_signals = st.multiselect(
        "Underlying model",
        ["LogReg_v92", "ElasticNet_v92", "Ensemble_v92"],
        default=["Ensemble_v92"],
    )

    st.subheader("Training Cut-Off")
    data_begin   = pd.Timestamp("2005-01-01").date()   # earliest possible OOS start
    data_latest  = (pd.Timestamp.today() - pd.DateOffset(years=3)).date()
    oos_start_dt = st.date_input(
        "OOS starts after (train ends on)",
        value=pd.Timestamp(TEST_START).date(),
        min_value=data_begin,
        max_value=data_latest,
        help=(
            "Walk-forward OOS begins from this date. "
            "Moving it earlier gives more OOS history; "
            "moving it later uses more data for training."
        ),
    )
    oos_start_str = oos_start_dt.strftime("%Y-%m-%d")
    st.caption(f"Walk-Forward OOS: **{oos_start_str}** → present")

    st.subheader("Date Range")
    date_min = oos_start_dt
    date_max = pd.Timestamp.today().date()
    date_start = st.date_input("From", value=date_min, min_value=date_min, max_value=date_max)
    date_end   = st.date_input("To",   value=date_max, min_value=date_min, max_value=date_max)

    st.divider()
    pos_model_sel = st.selectbox(
        "Position chart model",
        options=st.session_state.get("model_list", ["Ensemble_v92_SoftRisk_LO"]),
    )
    st.divider()
    st.caption(
        "Only strategies that **beat SPX B&H** (final NAV) are shown. "
        "P&L = pos_{t-1}×som_ret + pos_t×remaining_ret (full split formula)."
    )
    st.divider()
    if st.button("🔄 Force Refresh", type="secondary"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Cache: 1 hr · Pipeline: ~3 min first run")

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
data = load_all(oos_start_str)
backtest_df = data["backtest_df"]
current_df  = data["current_df"]
df_wf       = data["df_wf"]
df_v92      = data["df_v92"]
all_hist    = data["all_hist"]

# Update sidebar model list after data load
model_list = sorted(backtest_df["model_variant"].dropna().unique().tolist())
if st.session_state.get("model_list") != model_list:
    st.session_state["model_list"] = model_list

# ─────────────────────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────────────────────
ts_start = pd.Timestamp(date_start)
ts_end   = pd.Timestamp(date_end)

def _source_of(mv):
    """Return 'Combined', 'EOM', or 'Midmonth' based on model_variant name."""
    if "_Combined_" in mv: return "Combined"
    if "_EOM_"      in mv: return "EOM"
    return "Midmonth"

def mv_passes(mv):
    fam_ok = any(f in mv for f in sel_families)
    var_ok = any(f"_{v}" in mv for v in sel_variants)
    # strip EOM/Combined prefix for signal matching
    sig_ok = any(s.replace("_EOM","").replace("_Combined","") in mv for s in sel_signals)
    src_ok = _source_of(mv) in sel_sources
    return fam_ok and var_ok and sig_ok and src_ok

# Compute B&H NAV over full OOS window to identify underperformers
def _bh_nav_final(bdf, t0, t1):
    """Return final B&H NAV over [t0, t1] from first available model_variant."""
    for mv in bdf["model_variant"].unique():
        sub = bdf[(bdf["model_variant"] == mv) & (bdf.index >= t0) & (bdf.index <= t1)].sort_index()
        if len(sub) >= 2:
            return float((1 + sub["bh_pnl"] / 100).cumprod().iloc[-1])
    return 1.0

bh_nav_final = _bh_nav_final(backtest_df, ts_start, ts_end)

def _beats_bh(mv, bdf, t0, t1):
    sub = bdf[(bdf["model_variant"] == mv) & (bdf.index >= t0) & (bdf.index <= t1)].sort_index()
    if len(sub) < 3:
        return False
    strat_nav = float((1 + sub["strategy_pnl"] / 100).cumprod().iloc[-1])
    return strat_nav >= bh_nav_final

# Set of model_variants that beat B&H (used to filter all tables & figures)
all_mvs = backtest_df["model_variant"].unique()
beat_bh_mvs = {mv for mv in all_mvs if _beats_bh(mv, backtest_df, ts_start, ts_end)}

filt_backtest = backtest_df[
    backtest_df["model_variant"].apply(mv_passes) &
    backtest_df["model_variant"].isin(beat_bh_mvs) &
    (backtest_df.index >= ts_start) &
    (backtest_df.index <= ts_end)
].copy()

filt_current = current_df[
    current_df["family"].isin(sel_families) &
    current_df["variant"].isin(sel_variants) &
    current_df["signal"].str.replace("_EOM","",regex=False).isin(sel_signals) &
    current_df["model"].isin(beat_bh_mvs) &
    current_df["model"].apply(_source_of).isin(sel_sources)
].copy()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
latest_month = df_wf.index.max()
st.title("SPX Mid-Month Strategy v9.2")
st.caption(
    f"OOS: {oos_start_str} → {ts_end.strftime('%Y-%m-%d')}  |  "
    f"Latest signal month: **{latest_month.strftime('%Y-%m')}**  |  "
    f"Models shown: {len(filt_current)}"
)

# ─────────────────────────────────────────────────────────────
# ROW 1 — DECISION TABLE + RISK STATE
# ─────────────────────────────────────────────────────────────
col_dec, col_risk = st.columns([3, 2], gap="medium")

with col_dec:
    st.subheader("📋 Current Month Decision")

    disp_cols = [
        "signal", "family", "variant",
        "latest_logit", "latest_prob_up",
        "current_position", "after_position", "action",
        "sharpe_test", "nav_test",
    ]
    df_disp = filt_current[[c for c in disp_cols if c in filt_current.columns]].copy()
    df_disp = df_disp.rename(columns={
        "latest_logit":   "Logit",
        "latest_prob_up": "P(up)",
        "current_position": "Cur Pos",
        "after_position":   "New Pos",
        "sharpe_test":    "Sharpe",
        "nav_test":       "NAV",
    })

    def style_action(val):
        v = str(val)
        if "Increase" in v:  return "color:#3fb950;font-weight:700"
        if "Decrease" in v:  return "color:#f85149;font-weight:700"
        if v == "Hold":      return "color:#8b949e"
        return ""

    def style_sharpe(val):
        try:
            f = float(val)
            if f >= 2.5: return "background-color:#0d4f2e;color:#3fb950"
            if f >= 1.5: return "background-color:#1a3a1a"
            if f < 0:    return "background-color:#4f0d0d;color:#f85149"
        except: pass
        return ""

    styled = (
        df_disp.style
        .format({
            "Logit":   "{:+.3f}",
            "P(up)":   "{:.1%}",
            "Cur Pos": "{:.3f}",
            "New Pos": "{:.3f}",
            "Sharpe":  "{:.2f}",
            "NAV":     "{:.3f}",
        }, na_rep="—")
        .applymap(style_action, subset=["action"])
        .applymap(style_sharpe, subset=["Sharpe"])
    )
    st.dataframe(styled, use_container_width=True, height=min(60 + 35 * len(df_disp), 460))

with col_risk:
    st.subheader("🧭 Risk State Snapshot")
    latest_row = df_wf.iloc[-1]
    prev_row   = df_wf.iloc[-2] if len(df_wf) > 1 else df_wf.iloc[-1]

    def _get(row, c, default=np.nan):
        try:
            v = row[c] if c in row.index else default
            return float(v) if not pd.isna(v) else default
        except: return default

    risk_items = [
        ("Kalman Uncertainty", "kalman_uncertainty",   "{:.4f}", True),
        ("HMM Stress Prob",    "hmm_stress_prob",      "{:.1%}",  True),
        ("Hawkes VIX Intensity","hawkes_vix_intensity", "{:.3f}", True),
        ("VIX Regime (0-3)",   "vix_regime",           "{:.0f}",  True),
    ]
    rc1, rc2 = st.columns(2)
    for i, (label, col, fmt, inv) in enumerate(risk_items):
        val  = _get(latest_row, col)
        prev = _get(prev_row, col)
        dval = round(val - prev, 5) if (not np.isnan(val) and not np.isnan(prev)) else None
        disp = fmt.format(val) if not np.isnan(val) else "N/A"
        ddsp = (f"{dval:+.4f}" if dval is not None else None)
        with (rc1 if i % 2 == 0 else rc2):
            st.metric(label=label, value=disp, delta=ddsp,
                      delta_color="inverse" if inv else "normal")

    st.markdown("**Composite Risk Score**")
    unc = _get(latest_row, "kalman_uncertainty", 0)
    hmm = _get(latest_row, "hmm_stress_prob", 0.5)
    hk  = _get(latest_row, "hawkes_vix_intensity", 0)
    vr  = _get(latest_row, "vix_regime", 1) / 3.0
    composite = float(np.clip(
        0.20 * min(unc, 1.0) + 0.15 * hmm + 0.15 * min(hk / 3.0, 1.0) + 0.50 * vr, 0, 1
    ))

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(composite, 3),
        number={"font": {"size": 28, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e"}},
            "bar":  {"color": "#e3b341", "thickness": 0.25},
            "bgcolor": "#0e1117",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0.0, 0.3], "color": "#0d4f2e"},
                {"range": [0.3, 0.6], "color": "#3a2a00"},
                {"range": [0.6, 1.0], "color": "#4f0d0d"},
            ],
            "threshold": {"line": {"color": "#f85149", "width": 3},
                          "thickness": 0.75, "value": 0.65},
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=190, margin=dict(l=20, r=20, t=10, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# ROW 2 — NAV CURVES (full width)
# ─────────────────────────────────────────────────────────────
st.subheader("📈 Cumulative NAV — OOS")

COLOR_POOL = px.colors.qualitative.Dark24
fig_nav = go.Figure()

# B&H baseline from any variant
bh_added = False
for mv in filt_backtest["model_variant"].unique():
    sub = filt_backtest[filt_backtest["model_variant"] == mv].sort_index()
    if len(sub) < 2:
        continue
    if not bh_added:
        bh_nav = (1 + sub["bh_pnl"] / 100).cumprod()
        fig_nav.add_trace(go.Scatter(
            x=sub.index, y=bh_nav, name="SPX B&H",
            mode="lines", line=dict(color="#8b949e", width=2, dash="dot"),
        ))
        bh_added = True

for i, mv in enumerate(sorted(filt_backtest["model_variant"].unique())):
    sub = filt_backtest[filt_backtest["model_variant"] == mv].sort_index()
    if len(sub) < 2:
        continue
    nav = (1 + sub["strategy_pnl"] / 100).cumprod()
    fig_nav.add_trace(go.Scatter(
        x=sub.index, y=nav, name=mv, mode="lines",
        line=dict(color=COLOR_POOL[i % len(COLOR_POOL)], width=1.8),
        opacity=0.9,
        hovertemplate="<b>%{x|%Y-%m}</b><br>%{fullData.name}: %{y:.3f}<extra></extra>",
    ))

fig_nav.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
    xaxis_title="Month", yaxis_title="NAV (start = 1.0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    height=440,
    margin=dict(l=50, r=20, t=20, b=50),
    hovermode="x unified",
    xaxis=dict(showgrid=True, gridcolor="#21262d"),
    yaxis=dict(showgrid=True, gridcolor="#21262d"),
)
st.plotly_chart(fig_nav, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# ROW 3 — SUMMARY TABLE + POSITION HISTORY
# ─────────────────────────────────────────────────────────────
col_tbl, col_pos = st.columns([2, 3], gap="medium")

with col_tbl:
    st.subheader("📊 Backtest Summary")
    oos_ret = _safe(df_v92, "remaining_ret")
    rows = []
    for mv in sorted(filt_backtest["model_variant"].unique()):
        sub = filt_backtest[filt_backtest["model_variant"] == mv].sort_index()
        if len(sub) < 3:
            continue
        pnl = sub["strategy_pnl"]
        nav_v = float((1 + pnl / 100).cumprod().iloc[-1])
        sharpe_v = float(np.sqrt(12) * pnl.mean() / (pnl.std() + 1e-12))
        dd_arr = (1 + pnl / 100).cumprod()
        mdd_v  = float((dd_arr / dd_arr.cummax() - 1).min() * 100)
        pos_v  = sub["position"].abs().mean()
        rows.append({
            "Model":   mv,
            "Sharpe":  round(sharpe_v, 2),
            "NAV":     round(nav_v, 3),
            "MaxDD%":  round(mdd_v, 2),
            "AvgPos":  round(float(pos_v), 3),
        })

    if rows:
        df_sum = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)

        def _cs(v):
            try:
                f = float(v)
                if f >= 2.5: return "background-color:#0d4f2e;color:#3fb950"
                if f >= 1.5: return "background-color:#1a3a1a"
                if f < 0:    return "background-color:#4f0d0d;color:#f85149"
            except: pass
            return ""

        def _cd(v):
            try:
                f = float(v)
                if f <= -10: return "background-color:#4f0d0d;color:#f85149"
                if f <= -5:  return "background-color:#3a1a1a"
                if f >= -2:  return "background-color:#0d4f2e"
            except: pass
            return ""

        styled_sum = (
            df_sum.style
            .applymap(_cs, subset=["Sharpe"])
            .applymap(_cd, subset=["MaxDD%"])
            .format({"Sharpe": "{:.2f}", "NAV": "{:.3f}",
                     "MaxDD%": "{:.1f}%", "AvgPos": "{:.3f}"})
        )
        st.dataframe(styled_sum, use_container_width=True,
                     height=min(60 + 35 * len(df_sum), 500))
    else:
        st.info("No data matches current filters.")

with col_pos:
    st.subheader("📉 Position & NAV History")
    pos_mv = pos_model_sel

    sub_pos = backtest_df[
        (backtest_df["model_variant"] == pos_mv) &
        (backtest_df.index >= ts_start) &
        (backtest_df.index <= ts_end)
    ].sort_index()

    if len(sub_pos) > 0:
        bar_colors = sub_pos["position"].apply(
            lambda p: "#3fb950" if p > 0.05 else ("#f85149" if p < -0.05 else "#555e6a")
        )
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Bar(
            x=sub_pos.index, y=sub_pos["position"],
            marker_color=bar_colors,
            name="Position",
            hovertemplate="<b>%{x|%Y-%m}</b><br>Position: %{y:.3f}<extra></extra>",
        ))
        nav_line = (1 + sub_pos["strategy_pnl"] / 100).cumprod()
        fig_pos.add_trace(go.Scatter(
            x=sub_pos.index, y=nav_line,
            name="NAV", yaxis="y2",
            line=dict(color="#e3b341", width=2.5),
            mode="lines",
            hovertemplate="<b>%{x|%Y-%m}</b><br>NAV: %{y:.3f}<extra></extra>",
        ))
        fig_pos.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            title=dict(text=pos_mv, font=dict(size=13)),
            xaxis_title="Month",
            yaxis=dict(title="Position", range=[-1.15, 1.15],
                       showgrid=True, gridcolor="#21262d"),
            yaxis2=dict(title="NAV", overlaying="y", side="right",
                        showgrid=False, tickformat=".2f"),
            barmode="relative",
            height=420,
            legend=dict(orientation="h", y=1.04, font=dict(size=11)),
            margin=dict(l=50, r=60, t=45, b=50),
            hovermode="x unified",
        )
        st.plotly_chart(fig_pos, use_container_width=True)
    else:
        st.info("No data for selected model / date range.")

st.divider()

# ─────────────────────────────────────────────────────────────
# ROW 4 — DRAWDOWN CURVES
# ─────────────────────────────────────────────────────────────
st.subheader("📉 Drawdown Curves")
fig_dd = go.Figure()

dd_bh_added = False
for i, mv in enumerate(sorted(filt_backtest["model_variant"].unique())):
    sub = filt_backtest[filt_backtest["model_variant"] == mv].sort_index()
    if len(sub) < 2:
        continue
    if not dd_bh_added:
        bh_nav = (1 + sub["bh_pnl"] / 100).cumprod()
        bh_dd  = (bh_nav / bh_nav.cummax() - 1) * 100
        fig_dd.add_trace(go.Scatter(
            x=sub.index, y=bh_dd, name="SPX B&H",
            fill="tozeroy", mode="lines",
            line=dict(color="#8b949e", width=1.5, dash="dot"),
            fillcolor="rgba(139,148,158,0.10)",
        ))
        dd_bh_added = True
    nav = (1 + sub["strategy_pnl"] / 100).cumprod()
    dd  = (nav / nav.cummax() - 1) * 100
    fig_dd.add_trace(go.Scatter(
        x=sub.index, y=dd, name=mv, mode="lines",
        line=dict(color=COLOR_POOL[i % len(COLOR_POOL)], width=1.5),
        hovertemplate="<b>%{x|%Y-%m}</b><br>%{fullData.name}: %{y:.2f}%<extra></extra>",
    ))

fig_dd.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
    xaxis_title="Month", yaxis_title="Drawdown (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    height=340,
    margin=dict(l=50, r=20, t=20, b=50),
    hovermode="x unified",
    xaxis=dict(showgrid=True, gridcolor="#21262d"),
    yaxis=dict(showgrid=True, gridcolor="#21262d"),
)
st.plotly_chart(fig_dd, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ROW 5 — MONTHLY RETURN HEATMAP (best model by sharpe)
# ─────────────────────────────────────────────────────────────
if rows:
    best_mv = df_sum.iloc[0]["Model"]
    sub_best = backtest_df[
        (backtest_df["model_variant"] == best_mv) &
        (backtest_df.index >= ts_start) &
        (backtest_df.index <= ts_end)
    ].sort_index().copy()

    if len(sub_best) > 0:
        st.subheader(f"🗓 Monthly Return Heatmap — {best_mv}")
        # Reset index to avoid collision: backtest_df.index.name == "month"
        sub_best = sub_best.reset_index()
        date_col = sub_best.columns[0]   # the former index column (named "month" or similar)
        sub_best["_year"]  = pd.to_datetime(sub_best[date_col]).dt.year
        sub_best["_month"] = pd.to_datetime(sub_best[date_col]).dt.month

        pivot = sub_best.pivot_table(
            index="_year", columns="_month",
            values="strategy_pnl", aggfunc="sum"
        )
        pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index.astype(str),
            colorscale=[
                [0.0,  "#67001f"],
                [0.35, "#d6604d"],
                [0.50, "#161b22"],
                [0.65, "#4393c3"],
                [1.0,  "#053061"],
            ],
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="PnL %", tickfont=dict(color="#e6edf3")),
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed"),
            height=max(280, 35 * len(pivot) + 80),
            margin=dict(l=50, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

st.divider()
st.caption(
    f"SPX Mid-Month Strategy v9.2 · "
    f"Data: Stooq (SPX) + FRED (VIX, macro) · "
    f"Pipeline: Kalman / HMM / Hawkes / GARCH · "
    f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
)
