# streamlit_app.py  —  V9.4 SoftRisk edition
# Direct imports from spx_signal_v94: no subprocess, instant cache-based scoring.
import streamlit as st
import json, os, yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas_datareader as pdr

# ── Direct imports (eliminates subprocess overhead) ───────────────────────────
from spx_signal_v94 import (
    fetch_data, build_monthly, load_config,
    load_model_cache, train_and_cache_models, score_from_cache, cache_info,
    run_live_signal, run_backtest, load_pnl_history,
    decide, FEATURES, MODELS_CFG, ENSEMBLE_WEIGHTS,
    ENSEMBLE_CUTOFF, ENSEMBLE_SHORT_THR, SEASONAL_RULES,
    VIX_SPIKE_THRESH, _CACHE_PATH, SOFTRISK_CFG,
)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def get_edt_time():
    return datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d %H:%M:%S EDT')


DEFAULT_CONFIG = {
    "data_start":  "1990-01-01",
    "train_start": "1990-01-01",
    "train_end":   "2020-12-31",
    "test_start":  "2021-01-01",
    "notes":       "V9.4 SoftRisk strategy",
}

MODE_LABELS = {
    "soft_risk_lo": "🛡️ SoftRisk Long-Only (V9.4)",
    "soft_risk_ls": "⚡ SoftRisk Long-Short (V9.4)",
    "binary":       "🔵 Binary {0,1} (V9.1)",
}

st.set_page_config(page_title="SPX Mid-Month V9.4", layout="wide")


# ══════════════════════════════════════════════════════════════════════════════
# Cached data loading  (re-fetches at most once per hour)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="📡 Fetching market data...")
def get_monthly_df(data_start):
    cfg  = load_config()
    dfd, macro = fetch_data(data_start)
    df_m = build_monthly(dfd, macro)
    if cfg.get("train_start"):
        df_m = df_m[df_m.index >= cfg["train_start"]]
    return df_m


@st.cache_data(ttl=300, show_spinner="📊 Fetching real-time metrics...")
def get_realtime_metrics():
    end        = datetime.today()
    data_start = end - timedelta(days=90)

    vix_current = spx_latest = spx_chg_pct = np.nan
    vix_chg_pct = spx_volume_latest = volume_chg_pct = np.nan

    try:
        vix       = pdr.get_data_fred("VIXCLS", start=data_start, end=end)
        vix_close = vix["VIXCLS"].dropna()
        if len(vix_close) >= 40:
            vix_current = float(vix_close.tail(20).mean())
            vix_prev    = float(vix_close.tail(40).head(20).mean())
            vix_chg_pct = (vix_current - vix_prev) / vix_prev * 100 if vix_prev else np.nan
    except Exception as e:
        st.warning(f"VIX error: {e}")

    try:
        spx        = pdr.get_data_stooq("^SPX", start=data_start, end=end).sort_index()
        spx_close  = spx["Close"].dropna()
        spx_volume = spx["Volume"].dropna()
        if len(spx_close) >= 2:
            spx_latest  = float(spx_close.iloc[-1])
            spx_chg_pct = (spx_latest - float(spx_close.iloc[-2])) / float(spx_close.iloc[-2]) * 100
        if len(spx_volume) >= 6:
            spx_volume_latest = float(spx_volume.iloc[-1])
            avg               = float(spx_volume.tail(6).head(5).mean())
            volume_chg_pct    = (spx_volume_latest - avg) / avg * 100 if avg else np.nan
    except Exception as e:
        st.warning(f"SPX error: {e}")

    return {
        "vix_monthly_avg": round(vix_current, 2)       if pd.notna(vix_current)       else np.nan,
        "vix_chg_pct":     round(vix_chg_pct, 1)       if pd.notna(vix_chg_pct)       else np.nan,
        "spx_price":       round(spx_latest, 2)         if pd.notna(spx_latest)         else np.nan,
        "spx_chg_pct":     round(spx_chg_pct, 2)       if pd.notna(spx_chg_pct)       else np.nan,
        "spx_volume":      int(spx_volume_latest)       if pd.notna(spx_volume_latest) else np.nan,
        "volume_chg_pct":  round(volume_chg_pct, 1)    if pd.notna(volume_chg_pct)    else np.nan,
    }


def models_json_to_df(data: dict) -> pd.DataFrame:
    rows = []
    for name, info in data.get("models", {}).items():
        rows.append({
            "Model":    name,
            "Logit":    info.get("logit"),
            "P(up)":    info.get("prob_up"),
            "Signal":   info.get("signal"),
            "Position": info.get("position"),
            "Short Thr":info.get("short_thr"),
            "Month":    data.get("signal_month"),
            "VIX Spike":data.get("vix_spike"),
            "Mode":     data.get("mode"),
        })
    return pd.DataFrame(rows)


def live_signal_from_cache(df_m, mode, cfg):
    """
    Score current month using cached pre-trained models.
    Returns same dict structure as run_live_signal(as_json=False).
    """
    cache = load_model_cache()
    if cache is None:
        return None

    df_hist = df_m[df_m["remaining_ret"].notna()].copy()
    df_hist["y"] = (df_hist["remaining_ret"] > 0).astype(int)
    df_hist = df_hist[FEATURES + ["remaining_ret", "y"]].dropna()

    live_candidates = df_m[FEATURES].dropna()
    if len(live_candidates) == 0:
        return None
    live_row      = live_candidates.iloc[[-1]]
    current_date  = live_row.index[0]
    current_month = current_date.month

    vix_chg1m  = float(df_m["vix_chg1m"].iloc[-1]) if pd.notna(df_m["vix_chg1m"].iloc[-1]) else 0.0
    vix_spiked = vix_chg1m > VIX_SPIKE_THRESH
    vix_now    = float(df_m["vix_month"].iloc[-1])

    pnl_hist = load_pnl_history("EnsembleW", n=24)

    logits, probs, signals, positions = {}, {}, {}, {}
    for name, model_cfg in MODELS_CFG.items():
        logit, prob = score_from_cache(cache, name, live_row, FEATURES)
        sig, pos    = decide(logit, model_cfg["cutoff"], model_cfg["short_thr"],
                             current_month, vix_spiked, 0.0,
                             mode=mode, pnl_history=pnl_hist)
        logits[name], probs[name], signals[name], positions[name] = (
            round(logit, 4), round(prob, 4), sig, pos)

    ens_logit = round(sum(ENSEMBLE_WEIGHTS[n] * logits[n] for n in ENSEMBLE_WEIGHTS), 4)
    ens_sig, ens_pos = decide(ens_logit, ENSEMBLE_CUTOFF, ENSEMBLE_SHORT_THR,
                               current_month, vix_spiked, 0.0,
                               mode=mode, pnl_history=pnl_hist)
    logits["EnsembleW"]    = ens_logit
    probs["EnsembleW"]     = round(float(1/(1+np.exp(-ens_logit))), 4)
    signals["EnsembleW"]   = ens_sig
    positions["EnsembleW"] = ens_pos

    return {
        "generated_at":  pd.Timestamp.today().isoformat(),
        "signal_month":  current_date.strftime("%Y-%m"),
        "vix_month_avg": round(vix_now, 2),
        "vix_spike":     vix_spiked,
        "mode":          mode,
        "cache_used":    True,
        "models": {
            n: {"logit": logits[n], "prob_up": probs[n],
                "signal": signals[n], "position": positions[n],
                "cutoff":    MODELS_CFG.get(n, {}).get("cutoff", ENSEMBLE_CUTOFF),
                "short_thr": MODELS_CFG.get(n, {}).get("short_thr", ENSEMBLE_SHORT_THR)}
            for n in ["LogReg", "ElasticNet", "EnsembleW"]
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# Page layout
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔥 SPX Mid-Month Strategy V9.4")
st.markdown("**SoftRisk Long-Short / Long-Only | Instant Cache Scoring | Walk-Forward Backtest**")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Config display & edit
    st.subheader("📋 Current Config")
    config_yaml = yaml.dump(DEFAULT_CONFIG, default_flow_style=False, sort_keys=False)
    st.code(config_yaml, language="yaml")

    new_config = st.text_area("✏️ Edit Config:", value=config_yaml, height=200)
    if st.button("💾 Save Config", type="secondary"):
        try:
            DEFAULT_CONFIG.update(yaml.safe_load(new_config))
            st.success("✅ Config saved!"); st.rerun()
        except Exception:
            st.error("❌ Invalid YAML")

    st.divider()

    # ── Strategy mode selector ────────────────────────────────────────────────
    st.subheader("🎯 Strategy Mode")
    mode = st.selectbox(
        "Mode",
        options=list(MODE_LABELS.keys()),
        format_func=lambda x: MODE_LABELS[x],
        index=0,
        help=(
            "soft_risk_lo: continuous [0,+1], no shorting\n"
            "soft_risk_ls: continuous [-1,+1], allows short\n"
            "binary: original V9.1 {0,1} positions"
        )
    )

    st.divider()

    # ── Model Cache Management ────────────────────────────────────────────────
    st.subheader("🤖 Model Cache")
    cache = load_model_cache()

    if cache:
        st.success(f"✅ Cache ready\n`{cache_info(cache)}`")
    else:
        st.warning("⚠️ No cache — train models first for instant scoring")

    col_a, col_b = st.columns(2)
    with col_a:
        do_train_cache = st.button(
            "🏋️ Train & Cache" if not cache else "🔄 Retrain Cache",
            type="primary", use_container_width=True,
            help="Train on full history and save model_cache.pkl"
        )
    with col_b:
        if cache and st.button("🗑️ Clear Cache", use_container_width=True):
            if os.path.exists(_CACHE_PATH):
                os.remove(_CACHE_PATH)
            st.warning("Cache cleared"); st.rerun()

    if do_train_cache:
        with st.spinner("Training models on full history... (30-60s)"):
            try:
                cfg  = load_config()
                df_m = get_monthly_df(cfg["data_start"])
                df_hist = df_m[df_m["remaining_ret"].notna()].copy()
                df_hist["y"] = (df_hist["remaining_ret"] > 0).astype(int)
                df_hist = df_hist[FEATURES + ["remaining_ret","y"]].dropna()
                train_and_cache_models(df_hist, FEATURES)
                st.success(f"✅ Models cached! N={len(df_hist)} months"); st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    st.divider()
    st.caption(
        f"📌 SoftRisk defaults:\n"
        f"  scale={SOFTRISK_CFG['base_scale']}, "
        f"floor={SOFTRISK_CFG['floor']}\n"
        f"  λ_dd={SOFTRISK_CFG['lambda_dd']}, "
        f"λ_tail={SOFTRISK_CFG['lambda_tail']}\n"
        f"  shrink={SOFTRISK_CFG['shrink_mode']}"
    )


# ── Main columns ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📈 Current Month Signal")
    st.info(f"Mode: **{MODE_LABELS[mode]}**  |  "
            f"Cache: {'✅ ready' if cache else '⚠️ not built'}")

    use_cache_toggle = st.checkbox(
        "⚡ Use cached models (instant)", value=(cache is not None),
        disabled=(cache is None),
        help="Uncheck to retrain from scratch — takes ~30s"
    )

    if st.button("🔄 Compute Live Signal", type="primary"):
        with st.spinner("Scoring..."):
            try:
                cfg  = load_config()
                df_m = get_monthly_df(cfg["data_start"])

                if use_cache_toggle and cache:
                    live_data = live_signal_from_cache(df_m, mode, cfg)
                    latency   = "instant (cached)"
                else:
                    import io, contextlib
                    buf = io.StringIO()
                    logits, signals, positions = run_live_signal(
                        df_m, cfg, mode=mode, use_cache=False, as_json=False)
                    live_data = None   # already printed
                    latency   = "retrained"

                if live_data:
                    st.success(f"✅ Signal ready ({latency})")
                    df_sig = models_json_to_df(live_data)

                    # Colour position column
                    def color_pos(val):
                        if isinstance(val, float):
                            if val > 0.05:  return "color
