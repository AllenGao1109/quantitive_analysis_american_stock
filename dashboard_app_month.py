
"""
SPX Month Strategy — Interactive Dashboard
Run: streamlit run dashboard_app_month.py
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
from spx_month_strategy import (
    load_data, build_monthly, add_v92_features,
    get_bom_features, get_mid_features, walk_forward_generic,
    build_base_position_history, build_softrisk_position_history,
    build_logitstrength_position_history,
    compute_bom_pnl, compute_mid_pnl, compute_combined_pnl,
    summarize_strategy_stats, annualized_sharpe, max_drawdown,
    summarize_position_family, pick_top_ls_lo, build_cross_results,
    build_current_df, get_active_segment,
    SOFT_CFGS_BOM, SOFT_CFGS_MID, TEST_START, _safe,
)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="SPX Month Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="metric-container"] {
    background: #1e2630;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
}
.block-container { padding-top: 1.25rem; }
div[data-testid="stHorizontalBlock"] { gap: 0.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

COLOR_POOL = px.colors.qualitative.Dark24

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def model_variant_meta(mv: str) -> dict:
    if "COMBINED_TOP4_CROSS" in mv or mv.startswith("BOM["):
        return {"source": "Top4 Cross", "family": "Combined", "variant": "Mixed", "signal": "Cross"}
    if "CombinedBest__" in mv:
        return {"source": "Combined Best", "family": "Combined", "variant": "Mixed", "signal": "Best"}

    source = "BOM" if "_BOM_" in mv or mv.startswith("LogReg_v92_BOM") or mv.startswith("ElasticNet_v92_BOM") or mv.startswith("Ensemble_v92_BOM") else "MID"

    if "_SoftRisk_" in mv:
        family = "SoftRisk"
    elif "_Base_" in mv:
        family = "Base"
    elif "_LogitStr_" in mv:
        family = "LogitStrength"
    else:
        family = "Unknown"

    variant = "LS" if mv.endswith("_LS") else ("LO" if mv.endswith("_LO") else "Mixed")

    base = mv.split("_Base_")[0].split("_SoftRisk_")[0].split("_LogitStr_")[0]
    signal = base.replace("_BOM", "")

    return {"source": source, "family": family, "variant": variant, "signal": signal}


def current_model_meta(model: str) -> dict:
    if "_SoftRisk_" in model:
        family = "SoftRisk"
    elif "_Base_" in model:
        family = "Base"
    elif "_LogitStr_" in model:
        family = "LogitStrength"
    else:
        family = "Unknown"

    variant = "LS" if model.endswith("_LS") else ("LO" if model.endswith("_LO") else "Mixed")
    base = model.split("_Base_")[0].split("_SoftRisk_")[0].split("_LogitStr_")[0]
    signal = base.replace("_BOM", "")
    return {"family": family, "variant": variant, "signal": signal}


def build_backtest_df_oos(bom_hist, mid_hist, df_v92, best_bom_model, best_mid_model, df_cross, oos_start):
    parts = []

    for col in bom_hist.columns:
        df_pnl = compute_bom_pnl(_safe(bom_hist, col), df_v92, oos_start=oos_start)
        df_pnl["model_variant"] = col
        df_pnl["segment"] = "BOM"
        parts.append(df_pnl)

    for col in mid_hist.columns:
        df_pnl = compute_mid_pnl(_safe(mid_hist, col), df_v92, oos_start=oos_start)
        df_pnl["model_variant"] = col
        df_pnl["segment"] = "MID"
        parts.append(df_pnl)

    df_best = compute_combined_pnl(_safe(mid_hist, best_mid_model), _safe(bom_hist, best_bom_model), df_v92, oos_start=oos_start)
    df_best["model_variant"] = f"CombinedBest__BOM[{best_bom_model}]__MID[{best_mid_model}]"
    df_best["segment"] = "COMBINED_BEST"
    parts.append(df_best)

    if df_cross is not None and len(df_cross) > 0:
        for _, row in df_cross.iterrows():
            df_pair = compute_combined_pnl(_safe(mid_hist, row["mid_model"]), _safe(bom_hist, row["bom_model"]), df_v92, oos_start=oos_start)
            df_pair["model_variant"] = row["combined_name"]
            df_pair["segment"] = "COMBINED_TOP4_CROSS"
            parts.append(df_pair)

    backtest_df = pd.concat(parts).sort_index()
    backtest_df.index.name = "month"
    return backtest_df


def add_model_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    meta = out["model_variant"].apply(model_variant_meta)
    out["source"] = meta.apply(lambda d: d["source"])
    out["family"] = meta.apply(lambda d: d["family"])
    out["variant"] = meta.apply(lambda d: d["variant"])
    out["signal"] = meta.apply(lambda d: d["signal"])
    return out




def build_pnl_heatmap_df(pnl_series):
    df = pnl_series.copy().to_frame("pnl")
    df.index = pd.to_datetime(df.index)

    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot(index="year", columns="month", values="pnl")

    pivot = pivot.reindex(columns=range(1, 13))

    return pivot


def plot_pnl_heatmap(pivot_df, title="Monthly PnL Heatmap"):
    plt.figure(figsize=(10, 6))

    sns.heatmap(
        pivot_df,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "PnL (%)"}
    )

    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Year")

    return plt



@st.cache_data(ttl=3600, show_spinner="Running month strategy pipeline…")
def load_all(oos_start: str):
    df_daily, macro = load_data()
    df_m = build_monthly(df_daily, macro)
    df_v92 = add_v92_features(df_m)

    bom_features = get_bom_features(df_v92)
    mid_features = get_mid_features(df_v92)

    df_wf_bom = walk_forward_generic(df_v92, bom_features, target_col="y_bom", model_prefix="v92_BOM")
    df_wf_mid = walk_forward_generic(df_v92, mid_features, target_col="y_mid", model_prefix="v92")

    bom_sigs = [c for c in ["LogReg_v92_BOM", "ElasticNet_v92_BOM", "Ensemble_v92_BOM"] if c in df_wf_bom.columns]
    mid_sigs = [c for c in ["LogReg_v92", "ElasticNet_v92", "Ensemble_v92"] if c in df_wf_mid.columns]

    sig_hist_bom = df_wf_bom[bom_sigs].copy()
    sig_hist_mid = df_wf_mid[mid_sigs].copy()

    bom_hist = pd.concat([
        build_base_position_history(sig_hist_bom, base_scale=1.5),
        build_softrisk_position_history(sig_hist_bom, df_v92, SOFT_CFGS_BOM, tail_return_col="som_ret_bom", pnl_return_col="som_ret_bom"),
        build_logitstrength_position_history(sig_hist_bom),
    ], axis=1)
    bom_hist = bom_hist.loc[:, ~bom_hist.columns.duplicated(keep="first")]

    mid_hist = pd.concat([
        build_base_position_history(sig_hist_mid, base_scale=1.5),
        build_softrisk_position_history(sig_hist_mid, df_v92, SOFT_CFGS_MID, tail_return_col="remaining_ret", pnl_return_col="remaining_ret"),
        build_logitstrength_position_history(sig_hist_mid),
    ], axis=1)
    mid_hist = mid_hist.loc[:, ~mid_hist.columns.duplicated(keep="first")]

    bom_rank = summarize_position_family(bom_hist, df_v92, segment="bom")
    mid_rank = summarize_position_family(mid_hist, df_v92, segment="mid")
    best_bom_model = bom_rank.loc[0, "model"]
    best_mid_model = mid_rank.loc[0, "model"]

    bom_top4 = pick_top_ls_lo(bom_rank, n_ls=2, n_lo=2)
    mid_top4 = pick_top_ls_lo(mid_rank, n_ls=2, n_lo=2)
    df_cross = build_cross_results(bom_top4, mid_top4, bom_hist, mid_hist, df_v92)

    backtest_df = build_backtest_df_oos(bom_hist, mid_hist, df_v92, best_bom_model, best_mid_model, df_cross, oos_start=oos_start)
    backtest_df = add_model_meta_columns(backtest_df)

    current_df = build_current_df(
        df_daily=df_daily,
        df_wf_bom=df_wf_bom,
        df_wf_mid=df_wf_mid,
        bom_hist=bom_hist,
        mid_hist=mid_hist,
        df_v92=df_v92,
        split_day=12,
        oos_start=oos_start,
    )
    cur_meta = current_df["model"].apply(current_model_meta)
    current_df["variant"] = cur_meta.apply(lambda d: d["variant"])
    current_df["signal"] = cur_meta.apply(lambda d: d["signal"])

    active_segment, current_day_rank, latest_date = get_active_segment(df_daily, split_day=12)

    return {
        "df_daily": df_daily,
        "df_v92": df_v92,
        "df_wf_bom": df_wf_bom,
        "df_wf_mid": df_wf_mid,
        "bom_hist": bom_hist,
        "mid_hist": mid_hist,
        "bom_rank": bom_rank,
        "mid_rank": mid_rank,
        "bom_top4": bom_top4,
        "mid_top4": mid_top4,
        "df_cross": df_cross,
        "backtest_df": backtest_df,
        "current_df": current_df,
        "best_bom_model": best_bom_model,
        "best_mid_model": best_mid_model,
        "active_segment": active_segment,
        "current_day_rank": current_day_rank,
        "latest_date": latest_date,
    }

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.title("📈 SPX Month Strategy")
    st.divider()

    sel_families = st.multiselect(
        "Strategy family",
        ["Base", "SoftRisk", "LogitStrength", "Combined"],
        default=["SoftRisk"],
    )

    sel_variants = st.multiselect(
        "Direction",
        ["LO", "LS", "Mixed"],
        default=["LO", "LS"],
    )

    sel_sources = st.multiselect(
        "Source",
        ["BOM", "MID", "Combined Best", "Top4 Cross"],
        default=["BOM", "MID", "Top4 Cross"],
        help="BOM = start-of-month, MID = after day 12, Top4 Cross = 2 LS + 2 LO crossed",
    )

    sel_signals = st.multiselect(
        "Underlying model",
        ["LogReg_v92", "ElasticNet_v92", "Ensemble_v92", "Best", "Cross"],
        default=["Ensemble_v92"],
    )

    data_begin = pd.Timestamp("2005-01-01").date()
    data_latest = (pd.Timestamp.today() - pd.DateOffset(years=3)).date()
    oos_start_dt = st.date_input(
        "OOS starts",
        value=pd.Timestamp(TEST_START).date(),
        min_value=data_begin,
        max_value=data_latest,
    )
    oos_start_str = oos_start_dt.strftime("%Y-%m-%d")
    st.caption(f"Walk-forward OOS: **{oos_start_str}** → present")

    date_min = oos_start_dt
    date_max = pd.Timestamp.today().date()
    date_start = st.date_input("From", value=date_min, min_value=date_min, max_value=date_max)
    date_end = st.date_input("To", value=date_max, min_value=date_min, max_value=date_max)

    st.divider()
    pos_model_sel = st.selectbox(
        "Position chart model",
        options=st.session_state.get("model_list", ["Ensemble_v92_BOM_SoftRisk_LO"]),
    )

    st.divider()
    if st.button("🔄 Force Refresh", type="secondary"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Cache: 1 hr")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
data = load_all(oos_start_str)
backtest_df = data["backtest_df"]
current_df = data["current_df"]
df_v92 = data["df_v92"]
df_wf_bom = data["df_wf_bom"]
df_wf_mid = data["df_wf_mid"]
bom_hist = data["bom_hist"]
mid_hist = data["mid_hist"]
df_cross = data["df_cross"]
active_segment = data["active_segment"]
current_day_rank = data["current_day_rank"]
latest_date = data["latest_date"]

model_list = sorted(backtest_df["model_variant"].dropna().unique().tolist())
if st.session_state.get("model_list") != model_list:
    st.session_state["model_list"] = model_list

# ------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------
ts_start = pd.Timestamp(date_start)
ts_end = pd.Timestamp(date_end)

filt_backtest = backtest_df[
    backtest_df["source"].isin(sel_sources)
    & backtest_df["family"].isin(sel_families)
    & backtest_df["variant"].isin(sel_variants)
    & backtest_df["signal"].isin(sel_signals)
    & (backtest_df.index >= ts_start)
    & (backtest_df.index <= ts_end)
].copy()

filt_current = current_df[
    current_df["family"].isin([f for f in sel_families if f != "Combined"])
    & current_df["variant"].isin([v for v in sel_variants if v != "Mixed"])
    & current_df["signal"].isin([s for s in sel_signals if s not in ["Best", "Cross"]])
].copy()

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("SPX Month Strategy Dashboard")
st.caption(
    f"OOS: {oos_start_str} → {ts_end.strftime('%Y-%m-%d')}  |  "
    f"Active segment: **{active_segment}** (trade day **{current_day_rank}**)  |  "
    f"As of: **{pd.Timestamp(latest_date).strftime('%Y-%m-%d')}**"
)

# ------------------------------------------------------------
# ROW 1 — CURRENT DECISION + RISK
# ------------------------------------------------------------
col_dec, col_risk = st.columns([3, 2], gap="medium")

with col_dec:
    st.subheader("📋 Current Decision")
    disp_cols = [
        "segment", "model", "family",
        "latest_logit", "latest_prob_up",
        "target_position", "sharpe_test", "final_nav_test",
    ]
    df_disp = filt_current[[c for c in disp_cols if c in filt_current.columns]].copy()
    df_disp = df_disp.rename(columns={
        "latest_logit": "Logit",
        "latest_prob_up": "P(up)",
        "target_position": "Target Pos",
        "sharpe_test": "Sharpe",
        "final_nav_test": "NAV",
    })

    def style_sharpe(val):
        try:
            f = float(val)
            if f >= 2.5:
                return "background-color:#0d4f2e;color:#3fb950"
            if f >= 1.5:
                return "background-color:#1a3a1a"
            if f < 0:
                return "background-color:#4f0d0d;color:#f85149"
        except Exception:
            pass
        return ""

    styled = (
        df_disp.style
        .format({
            "Logit": "{:+.3f}",
            "P(up)": "{:.1%}",
            "Target Pos": "{:.3f}",
            "Sharpe": "{:.2f}",
            "NAV": "{:.3f}",
        }, na_rep="—")
        .applymap(style_sharpe, subset=["Sharpe"])
    )
    st.dataframe(styled, use_container_width=True, height=min(60 + 35 * max(len(df_disp), 1), 460))

with col_risk:
    st.subheader("🧭 Risk State Snapshot")
    source_wf = df_wf_bom if active_segment == "BOM" else df_wf_mid
    latest_row = source_wf.iloc[-1]
    prev_row = source_wf.iloc[-2] if len(source_wf) > 1 else source_wf.iloc[-1]

    def _get(row, c, default=np.nan):
        try:
            v = row[c] if c in row.index else default
            return float(v) if not pd.isna(v) else default
        except Exception:
            return default

    risk_items = [
        ("Kalman Uncertainty", "kalman_uncertainty", "{:.4f}", True),
        ("HMM Stress Prob", "hmm_stress_prob", "{:.1%}", True),
        ("Hawkes VIX Intensity", "hawkes_vix_intensity", "{:.3f}", True),
        ("VIX Regime (0-3)", "vix_regime", "{:.0f}", True),
    ]
    rc1, rc2 = st.columns(2)
    for i, (label, col, fmt, inv) in enumerate(risk_items):
        val = _get(latest_row, col)
        prev = _get(prev_row, col)
        dval = round(val - prev, 5) if (not np.isnan(val) and not np.isnan(prev)) else None
        disp = fmt.format(val) if not np.isnan(val) else "N/A"
        ddsp = (f"{dval:+.4f}" if dval is not None else None)
        with (rc1 if i % 2 == 0 else rc2):
            st.metric(label=label, value=disp, delta=ddsp, delta_color="inverse" if inv else "normal")

    unc = _get(latest_row, "kalman_uncertainty", 0)
    hmm = _get(latest_row, "hmm_stress_prob", 0.5)
    hk = _get(latest_row, "hawkes_vix_intensity", 0)
    vr = _get(latest_row, "vix_regime", 1) / 3.0
    composite = float(np.clip(0.20 * min(unc, 1.0) + 0.15 * hmm + 0.15 * min(hk / 3.0, 1.0) + 0.50 * vr, 0, 1))

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(composite, 3),
        number={"font": {"size": 28, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
            "bar": {"color": "#e3b341", "thickness": 0.25},
            "bgcolor": "#0e1117",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0.0, 0.3], "color": "#0d4f2e"},
                {"range": [0.3, 0.6], "color": "#3a2a00"},
                {"range": [0.6, 1.0], "color": "#4f0d0d"},
            ],
            "threshold": {"line": {"color": "#f85149", "width": 3}, "thickness": 0.75, "value": 0.65},
        },
    ))
    fig_gauge.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=190, margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ------------------------------------------------------------
# ROW 2 — NAV CURVES
# ------------------------------------------------------------
st.subheader("📈 Cumulative NAV")
fig_nav = go.Figure()

# benchmark from first visible series
bh_added = False
for mv in filt_backtest["model_variant"].unique():
    sub = filt_backtest[filt_backtest["model_variant"] == mv].sort_index()
    if len(sub) < 2:
        continue
    if not bh_added:
        bh_nav = (1 + sub["bh_pnl"] / 100).cumprod()
        fig_nav.add_trace(go.Scatter(
            x=sub.index, y=bh_nav, name="Benchmark",
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
    ))

fig_nav.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
    xaxis_title="Month", yaxis_title="NAV (start=1.0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    height=430, margin=dict(l=50, r=20, t=20, b=50), hovermode="x unified",
    xaxis=dict(showgrid=True, gridcolor="#21262d"), yaxis=dict(showgrid=True, gridcolor="#21262d"),
)
st.plotly_chart(fig_nav, use_container_width=True)

st.divider()

# ------------------------------------------------------------
# ROW 3 — SUMMARY + POSITION HISTORY
# ------------------------------------------------------------
col_tbl, col_pos = st.columns([2, 3], gap="medium")

with col_tbl:
    st.subheader("📊 Backtest Summary")
    rows = []
    for mv in sorted(filt_backtest["model_variant"].unique()):
        sub = filt_backtest[filt_backtest["model_variant"] == mv].sort_index()
        if len(sub) < 3:
            continue
        pnl = sub["strategy_pnl"]
        nav_v = float((1 + pnl / 100).cumprod().iloc[-1])
        sharpe_v = float(np.sqrt(12) * pnl.mean() / (pnl.std() + 1e-12))
        dd_arr = (1 + pnl / 100).cumprod()
        mdd_v = float((dd_arr / dd_arr.cummax() - 1).min() * 100)
        pos_v = sub["position"].abs().mean() if "position" in sub.columns else np.nan
        rows.append({
            "Model": mv,
            "Sharpe": round(sharpe_v, 2),
            "NAV": round(nav_v, 3),
            "MaxDD%": round(mdd_v, 2),
            "AvgPos": round(float(pos_v), 3) if pd.notna(pos_v) else np.nan,
        })

    if rows:
        df_sum = pd.DataFrame(rows).sort_values(["Sharpe", "NAV"], ascending=False).reset_index(drop=True)
        st.dataframe(df_sum, use_container_width=True, height=min(60 + 35 * len(df_sum), 520))
    else:
        st.info("No data matches current filters.")

with col_pos:
    st.subheader("📉 Position & NAV History")
    sub_pos = backtest_df[
        (backtest_df["model_variant"] == pos_model_sel)
        & (backtest_df.index >= ts_start)
        & (backtest_df.index <= ts_end)
    ].sort_index()

    if len(sub_pos) > 0:
        bar_colors = sub_pos["position"].apply(lambda p: "#3fb950" if p > 0.05 else ("#f85149" if p < -0.05 else "#555e6a"))
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Bar(x=sub_pos.index, y=sub_pos["position"], marker_color=bar_colors, name="Position"))
        nav_line = (1 + sub_pos["strategy_pnl"] / 100).cumprod()
        fig_pos.add_trace(go.Scatter(x=sub_pos.index, y=nav_line, name="NAV", yaxis="y2", line=dict(color="#e3b341", width=2.5), mode="lines"))
        fig_pos.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            title=dict(text=pos_model_sel, font=dict(size=13)),
            xaxis_title="Month",
            yaxis=dict(title="Position", range=[-1.15, 1.15], showgrid=True, gridcolor="#21262d"),
            yaxis2=dict(title="NAV", overlaying="y", side="right", showgrid=False, tickformat=".2f"),
            height=420, legend=dict(orientation="h", y=1.04, font=dict(size=11)), margin=dict(l=50, r=60, t=45, b=50),
        )
        st.plotly_chart(fig_pos, use_container_width=True)
    else:
        st.info("No data for selected model / date range.")

st.divider()

# ------------------------------------------------------------
# ROW 4 — DRAWDOWN + CROSS RESULTS
# ------------------------------------------------------------
col_dd, col_cross = st.columns([3, 2], gap="medium")

with col_dd:
    st.subheader("📉 Drawdown Curves")
    fig_dd = go.Figure()
    bh_added = False
    for i, mv in enumerate(sorted(filt_backtest["model_variant"].unique())):
        sub = filt_backtest[filt_backtest["model_variant"] == mv].sort_index()
        if len(sub) < 2:
            continue
        if not bh_added:
            bh_nav = (1 + sub["bh_pnl"] / 100).cumprod()
            bh_dd = (bh_nav / bh_nav.cummax() - 1) * 100
            fig_dd.add_trace(go.Scatter(x=sub.index, y=bh_dd, name="Benchmark DD", line=dict(color="#8b949e", width=2, dash="dot")))
            bh_added = True
        nav = (1 + sub["strategy_pnl"] / 100).cumprod()
        dd = (nav / nav.cummax() - 1) * 100
        fig_dd.add_trace(go.Scatter(x=sub.index, y=dd, name=mv, line=dict(color=COLOR_POOL[i % len(COLOR_POOL)], width=1.5)))

    fig_dd.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
        xaxis_title="Month", yaxis_title="Drawdown %",
        height=380, margin=dict(l=50, r=20, t=20, b=50), hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#21262d"), yaxis=dict(showgrid=True, gridcolor="#21262d"),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

with col_cross:
    st.subheader("🔀 Top4 × Top4 Cross")
    cross_view = df_cross.drop(columns=["combined_name"], errors="ignore").copy()
    st.dataframe(cross_view, use_container_width=True, height=380)

st.divider()


df_combined_sum = (
    backtest_df[backtest_df["segment"].astype(str).str.contains("COMBINED", na=False)]
    .groupby("model_variant", as_index=False)["strategy_pnl"]
    .agg(
        Sharpe=lambda x: annualized_sharpe(x),
        Final_NAV=lambda x: float((1 + pd.Series(x).fillna(0) / 100).cumprod().iloc[-1]),
        Avg_Monthly_PnL=lambda x: float(pd.Series(x).mean()),
    )
    .sort_values(["Sharpe", "Final_NAV"], ascending=False)
    .reset_index(drop=True)
)

best_combined_mv = df_combined_sum.iloc[0]["model_variant"]

if len(df_sum) > 0:
    best_mv = df_combined_sum.iloc[0]["model_variant"]
    sub_best = backtest_df[
        (backtest_df["model_variant"] == best_mv) &
        (backtest_df.index >= ts_start) &
        (backtest_df.index <= ts_end)
    ].sort_index().copy()

    if len(sub_best) > 0:
        st.subheader(f"🗓 Monthly Return Heatmap — {best_mv}")

        sub_best = sub_best.reset_index()
        date_col = sub_best.columns[0]

        sub_best["_year"] = pd.to_datetime(sub_best[date_col]).dt.year
        sub_best["_month"] = pd.to_datetime(sub_best[date_col]).dt.month

        pivot = sub_best.pivot_table(
            index="_year",
            columns="_month",
            values="strategy_pnl",
            aggfunc="sum"
        )

        pivot = pivot.reindex(columns=range(1, 13))
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
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed"),
            height=max(280, 35 * len(pivot) + 80),
            margin=dict(l=50, r=20, t=60, b=20),
        )

        st.plotly_chart(fig_heat, use_container_width=True)

st.divider()


# ------------------------------------------------------------
# FOOTER METRICS
# ------------------------------------------------------------
if len(df_cross) > 0:
    best_cross = df_cross.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Segment", active_segment)
    c2.metric("Top4 Cross Best Sharpe", f"{best_cross['sharpe']:.2f}")
    c3.metric("Top4 Cross Best NAV", f"{best_cross['final_nav']:.3f}")
    c4.metric("Top4 Cross Best MaxDD", f"{best_cross['max_dd']:.2f}%")
