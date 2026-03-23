import streamlit as st
import subprocess, json, os, yaml, pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from zoneinfo import ZoneInfo  
import pandas_datareader as pdr


def get_edt_time():
    edt = ZoneInfo("America/New_York")
    return datetime.now(edt).strftime('%Y-%m-%d %H:%M:%S EDT')



def models_json_to_df(data: dict) -> pd.DataFrame:
    models = data.get("models", {})

    rows = []
    for model_name, info in models.items():
        rows.append({
            "Model": model_name,
            "Logit": info.get("logit"),
            "P(up)": info.get("prob_up"),
            "Signal": info.get("signal"),
            "Position": info.get("position"),
            "Short Thr": info.get("short_thr"),
            "Signal Month": data.get("signal_month"),
            "VIX Spike": data.get("vix_spike"),
        })

    df = pd.DataFrame(rows)

    preferred_cols = [
        "Model", "Logit", "P(up)", "Signal",
        "Position", "Short Thr",
        "Signal Month", "VIX Spike"
    ]
    df = df[[c for c in preferred_cols if c in df.columns]]

    return df


def summary_to_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for idx, row in df_raw.iterrows():
        model_name = idx 

        summary = row["summary"]

        if isinstance(summary, str):
            summary = json.loads(summary)

        rows.append({
            "Model": model_name,
            "Alpha (cum)": summary.get("alpha_cum"),
            "Sharpe": summary.get("bh_sharpe"),
            "Calmar": summary.get("calmar"),
            "In Market %": summary.get("in_market"),
        })

    df = pd.DataFrame(rows)

    df["Alpha (cum)"] = pd.to_numeric(df["Alpha (cum)"], errors="coerce").round(2)
    df["Sharpe"] = pd.to_numeric(df["Sharpe"], errors="coerce").round(3)
    df["Calmar"] = pd.to_numeric(df["Calmar"], errors="coerce").round(3)
    df["In Market %"] = pd.to_numeric(df["In Market %"], errors="coerce").map(
        lambda x: f"{x:.1f}%" if pd.notnull(x) else ""
    )

    return df

st.set_page_config(page_title="SPX Mid-Month Strategy V9.1", layout="wide")

@st.cache_data(ttl=300)
def run_strategy(args):
    cmd = ["python", "spx_signal_v91.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", timeout=180)
    if result.returncode == 0:
        return json.loads(result.stdout.strip())
    return {"error": f"Exit {result.returncode}: {result.stderr[:500]}"}


# Get real-time metrics (latest data)
@st.cache_data(ttl=300)  # Updates every 5 minutes
def get_realtime_metrics():
    end = datetime.today()
    data_start = end - timedelta(days=90)

    # 初始化（默认 NaN）
    vix_current = float("nan")
    vix_chg_pct = float("nan")
    spx_latest = float("nan")
    spx_chg_pct = float("nan")
    spx_volume_latest = float("nan")
    volume_chg_pct = float("nan")
    sharpe_current = float("nan")
    sharpe_chg = float("nan")

    # ===================== VIX =====================
    try:
        vix = pdr.get_data_fred("VIXCLS", start=data_start, end=end)

        vix_close = vix["VIXCLS"].dropna()

        if len(vix_close) >= 40:
            vix_current = float(vix_close.tail(20).mean())
            vix_prev = float(vix_close.tail(40).head(20).mean())

            if vix_prev != 0:
                vix_chg_pct = ((vix_current - vix_prev) / vix_prev) * 100
    except Exception as e:
        st.warning(f"VIX error: {e}")

    # ===================== SPX =====================
    try:
        spx = pdr.get_data_stooq("^SPX", start=data_start, end=end).sort_index()

        spx_close = spx["Close"].dropna()
        spx_volume = spx["Volume"].dropna()

        if len(spx_close) >= 2:
            spx_latest = float(spx_close.iloc[-1])
            spx_prev = float(spx_close.iloc[-2])
            spx_chg_pct = ((spx_latest - spx_prev) / spx_prev) * 100

        if len(spx_volume) >= 6:
            spx_volume_latest = float(spx_volume.iloc[-1])
            spx_volume_avg = float(spx_volume.tail(6).head(5).mean())
            if spx_volume_avg != 0:
                volume_chg_pct = ((spx_volume_latest - spx_volume_avg) / spx_volume_avg) * 100

    except Exception as e:
        st.warning(f"SPX error: {e}")

    # ===================== Sharpe =====================
    try:
        bt_data = run_strategy(["--backtest", "--json"])
        sharpe_current = float(bt_data["summary"]["EnsembleW"]["sharpe"])
        sharpe_prev = float(bt_data["summary"]["LogReg"]["sharpe"])

        if sharpe_prev != 0:
            sharpe_chg = ((sharpe_current - sharpe_prev) / sharpe_prev) * 100

    except Exception as e:
        st.warning(f"Sharpe error: {e}")

    # ===================== RETURN =====================
    return {
        "vix_monthly_avg": round(vix_current, 2) if pd.notna(vix_current) else float("nan"),
        "vix_chg_pct": round(vix_chg_pct, 1) if pd.notna(vix_chg_pct) else float("nan"),
        "spx_price": round(spx_latest, 2) if pd.notna(spx_latest) else float("nan"),
        "spx_chg_pct": round(spx_chg_pct, 2) if pd.notna(spx_chg_pct) else float("nan"),
        "spx_volume": int(spx_volume_latest) if pd.notna(spx_volume_latest) else float("nan"),
        "volume_chg_pct": round(volume_chg_pct, 1) if pd.notna(volume_chg_pct) else float("nan"),
        "ensemble_sharpe": round(sharpe_current, 3) if pd.notna(sharpe_current) else float("nan"),
        "sharpe_chg_pct": round(sharpe_chg, 1) if pd.notna(sharpe_chg) else float("nan"),
    }


# Your default config
DEFAULT_CONFIG = {
    "data_start": "1990-01-01",
    "train_start": "1990-01-01", 
    "train_end": "2020-12-31",
    "test_start": "2021-01-01",
    "notes": "V9.1 ElasticNet+LogReg+EnsembleW strategy"
}

st.title("🔥 SPX Mid-Month Strategy V9.1")
st.markdown("**Live Signals + Full Backtest | Edit Config → Train New Models**")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Show current config
    st.subheader("📋 Current Config")
    config_yaml = yaml.dump(DEFAULT_CONFIG, default_flow_style=False, sort_keys=False)
    st.code(config_yaml, language="yaml")
    
    st.divider()
    
    # Edit config
    st.subheader("✏️ Edit Config (Save as New Default)")
    new_config = st.text_area(
        "Modify config.yaml:", 
        value=config_yaml,
        height=200,
        help="Edit parameters → Save → Run with new model"
    )
    
    if st.button("💾 Save as New Default Config", type="secondary"):
        try:
            DEFAULT_CONFIG.update(yaml.safe_load(new_config))
            st.success("✅ New default config saved!")
            st.rerun()
        except:
            st.error("❌ Invalid YAML format")
    
    st.divider()
    
    # Run options
    use_current_config = st.checkbox("✅ Use Current Config (Recommended)", value=True)
    
    if st.button("🚀 Run with Current Config", type="primary", disabled=use_current_config):
        st.session_state.custom_running = True
        st.rerun()
        
    st.caption("⚡ Config → Model → Signals (30-60s)")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📈 Current Month Signal")
    
    if use_current_config:
        st.info(f"✅ Using config: {DEFAULT_CONFIG['notes']}")
        if st.button("🔄 Compute Live Signal", type="secondary"):
            with st.spinner("Computing live signal with current config..."):
                live_data = run_strategy(["--json"])
                if "error" not in live_data:
                    st.success("✅ Signal computed!")
                    df_models = models_json_to_df(live_data)
    
                    st.dataframe(df_models)
                    st.caption(f"💾 Generated: {live_data.get('generated_at', 'Now')}")
                else:
                    st.error(f"❌ Error: {live_data['error']}")
    else:
        st.warning("👆 Select config and click 'Run with Current Config'")

with col2:
    st.header("📊 Backtest Results")
    
    if use_current_config:
        st.info(f"✅ Backtest period: {DEFAULT_CONFIG['test_start']} → Latest")
        if st.button("📊 Run Full Backtest", type="secondary"):
            with st.spinner("Running full backtest... (60s)"):
                bt_data = run_strategy(["--backtest", "--json"])
                if "error" not in bt_data:
                    st.success("✅ Backtest completed!")
                    if "summary" in bt_data:
                        df_summary = pd.DataFrame(bt_data["summary"]).T.reset_index()
                        df_summary = df_summary.rename(columns={"index": "Model"})
                        st.dataframe(df_summary, use_container_width=True, hide_index=False)
                else:
                    st.error(f"❌ Error: {bt_data['error']}")
    else:
        st.warning("👆 Select config and click 'Run with Current Config'")


# Footer metrics (ALL DYNAMIC)
st.markdown("---")

if "metrics_data" not in st.session_state:
    st.session_state.metrics_data = get_realtime_metrics()
    st.session_state.metrics_timestamp = get_edt_time()

col1, col2, col3, col4 = st.columns(4)
col_btn, col_time = st.columns([1, 5])

with col_btn:
    if st.button("🔄 Refresh Metrics", use_container_width=True):
        st.session_state.metrics_data = get_realtime_metrics()
        st.session_state.metrics_timestamp = get_edt_time()
        st.rerun()

with col_time:
    st.caption(f"🕐 Data snapshot time: {st.session_state.metrics_timestamp}")

metrics = st.session_state.metrics_data

# Dynamic deltas
delta_vix = f"{metrics['vix_chg_pct']:+.1f}%" if pd.notna(metrics["vix_chg_pct"]) else "-"
delta_spx = f"{metrics['spx_chg_pct']:+.2f}%" if pd.notna(metrics["spx_chg_pct"]) else "-"
delta_volume = f"{metrics['volume_chg_pct']:+.1f}%" if pd.notna(metrics["volume_chg_pct"]) else "-"
delta_sharpe = f"{metrics['sharpe_chg_pct']:+.1f}%" if pd.notna(metrics["sharpe_chg_pct"]) else "-"

vix_value = f"{metrics['vix_monthly_avg']}" if pd.notna(metrics["vix_monthly_avg"]) else "-"
spx_value = f"${metrics['spx_price']:,}" if pd.notna(metrics["spx_price"]) else "-"
sharpe_value = f"{metrics['ensemble_sharpe']}" if pd.notna(metrics["ensemble_sharpe"]) else "-"
volume_value = f"{int(metrics['spx_volume']):,}" if pd.notna(metrics["spx_volume"]) else "-"

col1.metric("VIX Monthly Avg", vix_value, delta_vix)
col2.metric("SPX Price", spx_value, delta_spx)
col3.metric("EnsembleW Sharpe", sharpe_value, delta_sharpe)
col4.metric("SPX Volume", volume_value, delta_volume)

st.caption(
    f"🕐 LIVE EDT: {get_edt_time()} | "
    f"Snapshot VIX: {vix_value} | SPX: {spx_value}"
)