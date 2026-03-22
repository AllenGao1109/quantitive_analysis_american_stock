import json, os
os.makedirs("output", exist_ok=True)
import sys, warnings, os, json, copy
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
import yaml 
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Load config
# ═══════════════════════════════════════════════════════════════════════════════
_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config():
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Hardcoded V9.1 model params (from trained pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
SEED             = 42
LOSS_WEIGHT_THR  = -2.0
LOSS_WEIGHT_MAX  = 5.0
VIX_SPIKE_THRESH = 5.0   # monthly: change in monthly-avg VIX
MIN_TRAIN        = 60

FEATURES = [
    "on_minus_id", "mom_d10", "overnight_avg", "id_ret_d10", "month_9",
    "vix_post_spike", "vol12m", "tbl_chg", "month_12", "tsmom",
    "dvp_lag1", "vol_d10", "on_ret_d10", "tom_early", "vix_chg3m"
]

MODELS_CFG = {
    "LogReg": dict(
        model="logreg", C=0.1, alpha=None, l1=None,
        cutoff=0.8, short_thr=0.6
    ),
    "ElasticNet": dict(
        model="elasticnet", C=None, alpha=0.005, l1=0.5,
        cutoff=1.2, short_thr=0.8
    ),
}
# EnsembleW fixed weights (from V9.1 rolling Sharpe avg: LR~42%, EN~58%)
ENSEMBLE_WEIGHTS   = {"LogReg": 0.42, "ElasticNet": 0.58}
ENSEMBLE_CUTOFF    = 1.1
ENSEMBLE_SHORT_THR = 0.9
SEASONAL_RULES     = {9: "sell", 1: "buy", 10: "buy", 12: "buy"}

os.makedirs("output", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Data fetching
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_data(data_start):
    end = "2026-12-31"
    spx = pdr.get_data_stooq("^SPX", start=data_start, end=end).sort_index()
    spx.index = pd.to_datetime(spx.index).tz_localize(None)
    spx.columns = [c.strip().title() for c in spx.columns]
    df = spx[["Open","High","Low","Close"]].dropna().copy()

    vix = pdr.get_data_fred("VIXCLS", start=data_start, end=end)
    vix.columns = ["vix"]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()

    df["daily_return"]  = df["Close"].pct_change() * 100
    df["overnight_ret"] = (df["Open"] / df["Close"].shift(1) - 1) * 100
    df["intraday_ret"]  = (df["Close"] / df["Open"] - 1) * 100
    df["yearmonth"]     = df.index.to_period("M")
    df["day_rank"]      = df.groupby("yearmonth").cumcount() + 1
    df["day_rank_rev"]  = (df.groupby("yearmonth")["Close"]
                            .transform("count") - df["day_rank"] + 1)
    df["high_252"]      = df["Close"].rolling(252, min_periods=126).max().shift(1)
    df["dist_52w"]      = (df["Close"] - df["high_252"]) / df["high_252"] * 100

    S, E = data_start, end
    tbl = pdr.get_data_fred("TB3MS",    start=S, end=E)
    lty = pdr.get_data_fred("GS10",     start=S, end=E)
    fed = pdr.get_data_fred("FEDFUNDS", start=S, end=E)
    baa = pdr.get_data_fred("BAA",      start=S, end=E)
    aaa = pdr.get_data_fred("AAA",      start=S, end=E)
    for d, n in [(tbl,"tbl"),(lty,"lty"),(fed,"fedfunds"),(baa,"baa"),(aaa,"aaa")]:
        d.columns = [n]
        d.index   = pd.to_datetime(d.index).to_period("M").to_timestamp()
    dfy = (baa["baa"] - aaa["aaa"]).to_frame("dfy")
    dfr = (lty["lty"]  - tbl["tbl"]).to_frame("dfr")
    macro = tbl.join([lty, fed, dfy, dfr], how="outer")
    macro["tbl_chg"]      = macro["tbl"].diff()
    macro["lty_chg"]      = macro["lty"].diff()
    macro["rate_rising"]  = (macro["fedfunds"].diff() >  0.1).astype(int)
    macro["rate_cutting"] = (macro["fedfunds"].diff() < -0.1).astype(int)
    macro["inverted"]     = (macro["dfr"] < 0).astype(int)
    macro.index = macro.index.to_period("M").to_timestamp()
    return df, macro


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Monthly feature builder
# ═══════════════════════════════════════════════════════════════════════════════
def build_monthly(df_daily, macro):
    month_groups   = {p: g.sort_index() for p, g in df_daily.groupby("yearmonth")}
    sorted_periods = sorted(month_groups.keys())
    records = []

    for i, period in enumerate(sorted_periods):
        group = month_groups[period]
        ts    = period.to_timestamp()
        c0    = group["Close"].iloc[0]
        monthly_ret   = (group["Close"].iloc[-1] / c0 - 1) * 100
        svar          = (group["daily_return"] ** 2).sum()
        vix_avg       = group["vix"].mean()
        rvm           = group["daily_return"].std() ** 2 * len(group)
        vrp           = vix_avg**2 / 100**2 * 12 - rvm / 10000
        overnight_avg = group["overnight_ret"].mean()
        intraday_avg  = group["intraday_ret"].mean()
        on_minus_id   = overnight_avg - intraday_avg
        overnight_vol = group["overnight_ret"].std()
        intraday_vol  = group["intraday_ret"].std()

        d13    = group[group["day_rank"] <= 3]
        d_last = group[group["day_rank_rev"] <= 1]
        tom_early = ((d13["Close"].iloc[-1]/d13["Close"].iloc[0]-1)*100
                     if len(d13) >= 2 else np.nan)
        tom_last  = d_last["daily_return"].iloc[0] if len(d_last) >= 1 else np.nan

        d8_12 = group[(group["day_rank"] >= 8) & (group["day_rank"] <= 12)]
        if len(d8_12) > 0:
            c_mid       = d8_12["Close"].mean()
            vix_d10     = d8_12["vix"].mean()
            dist52_d10  = (d8_12["dist_52w"].mean()
                           if d8_12["dist_52w"].notna().any() else np.nan)
            first_mid   = group[group["day_rank"] <= d8_12["day_rank"].max()]
            rv_d10      = first_mid["daily_return"].std()**2 * len(first_mid)
            vrp_d10     = vix_d10**2/100**2*12 - rv_d10/10000
            vol_d10     = first_mid["daily_return"].std()
            upr_d10     = (first_mid["daily_return"] > 0).mean()
            on_d10      = first_mid["overnight_ret"].mean()
            id_d10      = first_mid["intraday_ret"].mean()
            som_ret     = (c_mid / c0 - 1) * 100
            neg_rets    = first_mid["daily_return"][first_mid["daily_return"] < 0]
            dvp         = float(np.sqrt((neg_rets**2).mean())) if len(neg_rets) > 0 else 0.0
        else:
            c_mid = vix_d10 = dist52_d10 = vrp_d10 = vol_d10 = upr_d10 = np.nan
            on_d10 = id_d10 = som_ret = dvp = np.nan

        # remaining_ret: mid -> end of month (using last 2 days this + first 2 next)
        d_eom_this = group[group["day_rank_rev"] <= 2]["Close"]
        if i + 1 < len(sorted_periods):
            ng         = month_groups[sorted_periods[i+1]]
            d_eom_next = ng[ng["day_rank"] <= 2]["Close"]
        else:
            d_eom_next = pd.Series(dtype=float)
        eom_closes = pd.concat([d_eom_this, d_eom_next])
        if len(eom_closes) > 0 and pd.notna(c_mid):
            rem_ret = (eom_closes.mean() / c_mid - 1) * 100
        else:
            rem_ret = np.nan   # current/latest month: unknown until EOM

        mid     = group[(group["day_rank"] >= 7) & (group["day_rank"] <= 12)]
        vix_mid = mid["vix"].mean() if len(mid) > 0 else np.nan

        records.append(dict(
            month=ts,
            monthly_return=round(monthly_ret, 4),
            som_ret=round(som_ret, 4)         if pd.notna(som_ret)       else np.nan,
            remaining_ret=round(rem_ret, 4)   if pd.notna(rem_ret)       else np.nan,
            svar=round(svar, 6),
            vrp=round(vrp, 6),
            vrp_d10=round(vrp_d10, 6)         if pd.notna(vrp_d10)       else np.nan,
            vix_month=round(vix_avg, 4),
            vix_day10=round(vix_d10, 4)        if pd.notna(vix_d10)      else np.nan,
            vix_mid=round(vix_mid, 4)          if pd.notna(vix_mid)      else np.nan,
            overnight_avg=round(overnight_avg, 4),
            intraday_avg=round(intraday_avg, 4),
            on_minus_id=round(on_minus_id, 4),
            overnight_vol=round(overnight_vol, 4),
            intraday_vol=round(intraday_vol, 4),
            tom_early=round(tom_early, 4)      if pd.notna(tom_early)    else np.nan,
            tom_last=round(tom_last, 4)        if pd.notna(tom_last)     else np.nan,
            dist_52w=round(dist52_d10, 4)      if pd.notna(dist52_d10)   else np.nan,
            on_ret_d10=round(on_d10, 4)        if pd.notna(on_d10)       else np.nan,
            id_ret_d10=round(id_d10, 4)        if pd.notna(id_d10)       else np.nan,
            vol_d10=round(vol_d10, 4)          if pd.notna(vol_d10)      else np.nan,
            up_ratio_d10=round(upr_d10, 4)     if pd.notna(upr_d10)      else np.nan,
            dvp=round(dvp, 4)                  if pd.notna(dvp)          else np.nan,
            mom_d10=round(som_ret, 4)          if pd.notna(som_ret)      else np.nan,
        ))

    df_m = pd.DataFrame(records).set_index("month")
    df_m.index = df_m.index.to_period("M").to_timestamp()

    # Derived features
    df_m["rev1m"]          = df_m["monthly_return"].shift(1)
    df_m["mom3m"]          = df_m["monthly_return"].rolling(3).sum().shift(1)
    df_m["mom6m"]          = df_m["monthly_return"].rolling(6).sum().shift(2)
    df_m["mom12m_skip1"]   = df_m["monthly_return"].rolling(12).sum().shift(2)
    df_m["mom24m"]         = df_m["monthly_return"].rolling(24).sum().shift(2)
    df_m["vol6m"]          = df_m["monthly_return"].rolling(6).std().shift(1)
    df_m["vol12m"]         = df_m["monthly_return"].rolling(12).std().shift(1)
    df_m["vix_zscore12"]   = ((df_m["vix_month"] - df_m["vix_month"].rolling(12).mean())
                               / df_m["vix_month"].rolling(12).std())
    df_m["vix_chg1m"]      = df_m["vix_month"].diff(1)
    df_m["vix_chg3m"]      = df_m["vix_month"].diff(3)
    df_m["vix_spike"]      = (df_m["vix_chg1m"] > VIX_SPIKE_THRESH).astype(int)
    df_m["vix_post_spike"] = df_m["vix_spike"].shift(1).fillna(0).astype(int)
    df_m["vrp_x_vol12"]    = df_m["vrp"] * df_m["vol12m"].shift(1)
    df_m["tsmom"]          = df_m["mom12m_skip1"] / (df_m["vol12m"].shift(1) + 1e-8)
    df_m["dvp_lag1"]       = df_m["dvp"].shift(1)
    for moy in [1, 9, 10, 12]:
        df_m[f"month_{moy}"] = (df_m.index.month == moy).astype(int)

    macro.index = macro.index.to_period("M").to_timestamp()
    df_m = df_m.join(macro, how="left")
    for c in macro.columns:
        df_m[c] = df_m[c].ffill()

    df_m["macro_stress"]  = ((df_m["vix_zscore12"] > 1).astype(int) *
                              df_m["inverted"].astype(int) *
                              (df_m["lty_chg"] > 0.2).astype(int))
    df_m["stress_roll3"]  = df_m["macro_stress"].rolling(3, min_periods=1).mean()
    df_m["mom_x_stress"]  = df_m["mom12m_skip1"] * df_m["macro_stress"]
    return df_m


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Core: train + score one observation
# ═══════════════════════════════════════════════════════════════════════════════
def make_sample_weights(ret):
    w    = np.ones(len(ret))
    mask = ret < LOSS_WEIGHT_THR
    if mask.any():
        sev     = np.clip(np.abs(ret[mask] - LOSS_WEIGHT_THR) / abs(LOSS_WEIGHT_THR), 0, 1)
        w[mask] = 1.0 + (LOSS_WEIGHT_MAX - 1.0) * sev
    return w


def train_and_score(train_df, live_row, features, cfg):
    """
    Train on train_df (all rows with known remaining_ret),
    score live_row (current month, no remaining_ret required).
    Returns (logit, prob_up).
    """
    X_tr  = train_df[features].values
    y_tr  = train_df["y"].values
    w     = make_sample_weights(train_df["remaining_ret"].values)
    X_new = live_row[features].values

    sc = StandardScaler().fit(X_tr)
    np.random.seed(SEED)

    if cfg["model"] == "logreg":
        clf = LogisticRegression(C=cfg["C"], max_iter=2000, random_state=SEED)
        clf.fit(sc.transform(X_tr), y_tr, sample_weight=w)
    else:
        clf = SGDClassifier(loss="log_loss", penalty="elasticnet",
                            alpha=cfg["alpha"], l1_ratio=cfg["l1"],
                            max_iter=2000, random_state=SEED, tol=1e-4)
        clf.fit(sc.transform(X_tr), y_tr, sample_weight=w)

    prob  = float(np.clip(clf.predict_proba(sc.transform(X_new))[0, 1], 0.01, 0.99))
    logit = float(np.log(prob / (1 - prob)))
    return logit, prob


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Signal decision logic
# ═══════════════════════════════════════════════════════════════════════════════
def decide(logit, cutoff, short_thr, month, vix_spiked, prev_pos):
    if vix_spiked:
        return "VIX-EXIT", 0.0
    if logit > cutoff:
        return "BUY", 1.0
    elif logit < short_thr:
        return "SELL", 0.0
    else:
        # HOLD: check seasonal override
        rule = SEASONAL_RULES.get(month)
        if rule == "sell":
            return "SELL (seasonal)", 0.0
        elif rule == "buy":
            return "BUY (seasonal)", 1.0
        return "HOLD", float(prev_pos)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Position state (for live use)
# ═══════════════════════════════════════════════════════════════════════════════
_POS_PATH = os.path.join(os.path.dirname(__file__), "current_position.json")

def load_prev_position():
    try:
        with open(_POS_PATH) as f:
            d = json.load(f)
        return {k: float(v) for k, v in d.get("positions", {}).items()}
    except Exception:
        return {"LogReg": 0.0, "ElasticNet": 0.0, "EnsembleW": 0.0}

def save_position(positions, signals, logits):
    d = {
        "generated_at": pd.Timestamp.today().isoformat(),
        "positions": positions,
        "signals":   signals,
        "logits":    {k: round(v, 4) for k, v in logits.items()},
    }
    with open(_POS_PATH, "w") as f:
        json.dump(d, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Live signal (single month)
# ═══════════════════════════════════════════════════════════════════════════════
def run_live_signal(df_m, cfg_data, as_json=False):
    """
    df_m : full monthly feature table (including current month as last row,
           which has NaN remaining_ret -- that is expected and correct).
    """
    train_end  = cfg_data["train_end"]

    # Training data = all rows where remaining_ret is known AND within train_end
    # For live: use ALL history with known remaining_ret (extends beyond train_end)
    df_hist = df_m[df_m["remaining_ret"].notna()].copy()
    df_hist["y"] = (df_hist["remaining_ret"] > 0).astype(int)
    df_hist = df_hist[FEATURES + ["remaining_ret", "y"]].dropna()

    # Current month = last row of df_m (remaining_ret may be NaN -- live month)
    # Use the latest row that has all FEATURES available
    live_candidates = df_m[FEATURES].dropna()
    if len(live_candidates) == 0:
        raise ValueError("No complete feature row available for current month.")
    live_row = live_candidates.iloc[[-1]]
    current_date  = live_row.index[0]
    current_month = current_date.month

    # VIX spike: month-level definition (consistent with training)
    vix_chg1m     = float(df_m["vix_chg1m"].iloc[-1]) if pd.notna(df_m["vix_chg1m"].iloc[-1]) else 0.0
    vix_spiked    = vix_chg1m > VIX_SPIKE_THRESH
    vix_now       = float(df_m["vix_month"].iloc[-1])

    prev_pos = load_prev_position()
    logits, probs, signals, positions = {}, {}, {}, {}

    for name, model_cfg in MODELS_CFG.items():
        logit, prob = train_and_score(df_hist, live_row, FEATURES, model_cfg)
        sig, pos    = decide(logit, model_cfg["cutoff"], model_cfg["short_thr"],
                             current_month, vix_spiked, prev_pos.get(name, 0.0))
        logits[name]    = round(logit, 4)
        probs[name]     = round(prob, 4)
        signals[name]   = sig
        positions[name] = pos

    # EnsembleW: weighted logit
    ens_logit = round(sum(ENSEMBLE_WEIGHTS[n] * logits[n]
                          for n in ENSEMBLE_WEIGHTS), 4)
    ens_sig, ens_pos = decide(ens_logit, ENSEMBLE_CUTOFF, ENSEMBLE_SHORT_THR,
                               current_month, vix_spiked,
                               prev_pos.get("EnsembleW", 0.0))
    logits["EnsembleW"]    = ens_logit
    probs["EnsembleW"]     = round(float(1 / (1 + np.exp(-ens_logit))), 4)
    signals["EnsembleW"]   = ens_sig
    positions["EnsembleW"] = ens_pos

    save_position(positions, signals, logits)

    MODEL_NAMES = ["LogReg", "ElasticNet", "EnsembleW"]

    if as_json:
        out = {
            "generated_at":  pd.Timestamp.today().isoformat(),
            "signal_month":  current_date.strftime("%Y-%m"),
            "vix_month_avg": round(vix_now, 2),
            "vix_spike":     vix_spiked,
            "models": {
                name: {
                    "logit":     logits[name],
                    "prob_up":   probs[name],
                    "signal":    signals[name],
                    "position":  positions[name],
                    "cutoff":    MODELS_CFG.get(name, {}).get("cutoff", ENSEMBLE_CUTOFF),
                    "short_thr": MODELS_CFG.get(name, {}).get("short_thr", ENSEMBLE_SHORT_THR),
                }
                for name in MODEL_NAMES
            }
        }
        print(json.dumps(out, indent=2))
    else:
        W = 60
        arrow = lambda p: " ▲" if p == 1.0 else " ▼"
        print("=" * W)
        print(f"  SPX Mid-Month Signal  |  V9.1  |  {current_date.strftime('%Y-%m')}")
        print("=" * W)
        print(f"  VIX (monthly avg): {vix_now:.1f}  |  "
              f"spike: {'YES ⚠' if vix_spiked else 'no'}")
        print("-" * W)
        print(f"  {'Model':<14} {'Logit':>8} {'P(up)':>7}  {'Signal':<24} {'Pos'}")
        print("-" * W)
        for name in MODEL_NAMES:
            print(f"  {name:<14} {logits[name]:>+8.4f} {probs[name]:>6.1%}  "
                  f"{signals[name]:<24} {arrow(positions[name])}")
        print("=" * W)
        print(f"  Thresholds:")
        print(f"    LogReg:     BUY > {MODELS_CFG['LogReg']['cutoff']} | "
              f"SELL < {MODELS_CFG['LogReg']['short_thr']}")
        print(f"    ElasticNet: BUY > {MODELS_CFG['ElasticNet']['cutoff']} | "
              f"SELL < {MODELS_CFG['ElasticNet']['short_thr']}")
        print(f"    EnsembleW:  BUY > {ENSEMBLE_CUTOFF} | SELL < {ENSEMBLE_SHORT_THR}")
        print(f"  EnsembleW weights: "
              f"LogReg={ENSEMBLE_WEIGHTS['LogReg']:.0%}  "
              f"ElasticNet={ENSEMBLE_WEIGHTS['ElasticNet']:.0%}")
        print(f"  Seasonal: Jan/Oct/Dec=BUY  Sep=SELL  (when HOLD only)")
        print("=" * W)

    return logits, signals, positions


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Backtest (walk-forward, identical logic to live)
# ═══════════════════════════════════════════════════════════════════════════════
def run_backtest(df_m, cfg_data, as_json=False):
    test_start = cfg_data.get("test_start")
    train_end  = cfg_data["train_end"]

    if not test_start:
        msg = "test_start is null in config.json -- backtest skipped."
        if as_json:
            print(json.dumps({"skipped": True, "reason": msg}))
        else:
            print(msg)
        return None

    df_full = df_m.copy()
    df_full["y"] = (df_full["remaining_ret"] > 0).astype(int)
    df_avail = df_full[FEATURES + ["remaining_ret","som_ret",
                                    "monthly_return","vix_spike","y"]].dropna().copy()

    oos_idx  = df_avail[df_avail.index >= test_start].index
    if len(oos_idx) == 0:
        print("No OOS data found after test_start.")
        return None

    MODEL_NAMES = list(MODELS_CFG.keys()) + ["EnsembleW"]
    records     = {n: [] for n in MODEL_NAMES}
    positions   = {n: 0.0 for n in MODEL_NAMES}

    total = len(oos_idx)
    if not as_json:
        print(f"\nBacktest: {test_start} -> latest  ({total} months)")
        print("Progress: ", end="", flush=True)

    for i, date in enumerate(oos_idx):
        if not as_json and i % max(1, total//10) == 0:
            print(f"{i/total*100:.0f}%..", end="", flush=True)

        # Training: all rows strictly before this date with known remaining_ret
        train = df_avail[df_avail.index < date].copy()
        if len(train) < MIN_TRAIN:
            continue

        live_row = df_avail.loc[[date]]
        row      = df_avail.loc[date]
        s        = float(row["som_ret"])
        r        = float(row["remaining_ret"])
        bh       = float(row["monthly_return"])
        vix_spiked = bool(row["vix_spike"])
        month    = date.month

        # Score each model
        logits_this = {}
        for name, model_cfg in MODELS_CFG.items():
            logit, _ = train_and_score(train, live_row, FEATURES, model_cfg)
            logits_this[name] = logit

        logits_this["EnsembleW"] = sum(ENSEMBLE_WEIGHTS[n] * logits_this[n]
                                       for n in ENSEMBLE_WEIGHTS)

        thresholds = {
            "LogReg":     (MODELS_CFG["LogReg"]["cutoff"],
                           MODELS_CFG["LogReg"]["short_thr"]),
            "ElasticNet": (MODELS_CFG["ElasticNet"]["cutoff"],
                           MODELS_CFG["ElasticNet"]["short_thr"]),
            "EnsembleW":  (ENSEMBLE_CUTOFF, ENSEMBLE_SHORT_THR),
        }

        for name in MODEL_NAMES:
            logit      = logits_this[name]
            cutoff, st = thresholds[name]
            prev_pos   = positions[name]
            sig, new_pos = decide(logit, cutoff, st, month, vix_spiked, prev_pos)

            # PnL: SOM return on position held from prev month, then remaining_ret on new pos
            pnl = s * prev_pos + r * new_pos

            records[name].append(dict(
                date=date,
                logit=round(logit, 4),
                signal=sig,
                pos_start=prev_pos,
                pos_end=new_pos,
                som_ret=round(s, 4),
                remaining_ret=round(r, 4),
                strategy_pnl=round(pnl, 4),
                bh_pnl=round(bh, 4),
                alpha_pnl=round(pnl - bh, 4),
            ))
            positions[name] = new_pos

    if not as_json:
        print(" done.\n")

    # Metrics
    def calc_metrics(recs):
        df_r = pd.DataFrame(recs).set_index("date")
        s    = df_r["strategy_pnl"]
        bh   = df_r["bh_pnl"]
        nav  = (1 + s/100).cumprod()
        dd   = (nav/nav.cummax()-1).min()*100
        ann  = nav.iloc[-1]**(12/len(nav)) - 1
        sh   = s.mean()/(s.std()+1e-8)*np.sqrt(12)
        cal  = ann*100/abs(dd) if dd != 0 else np.nan
        inm  = (df_r["pos_end"] > 0).mean()*100
        bh_nav = (1 + bh/100).cumprod()
        bh_sh  = bh.mean()/(bh.std()+1e-8)*np.sqrt(12)
        return dict(sharpe=round(sh,3), nav=round(nav.iloc[-1],2),
                    alpha_cum=round((s-bh).sum(),1),
                    max_dd=round(dd,2), calmar=round(cal,3),
                    in_market=round(inm,1),
                    bh_sharpe=round(bh_sh,3), bh_nav=round(bh_nav.iloc[-1],2))

    summary = {n: calc_metrics(records[n]) for n in MODEL_NAMES}

    # Year-by-year
    yby = {}
    for name in MODEL_NAMES:
        df_r = pd.DataFrame(records[name]).set_index("date")
        yby[name] = {str(yr): round(df_r[df_r.index.year==yr]["strategy_pnl"].sum(),2)
                     for yr in range(df_r.index.year.min(), df_r.index.year.max()+1)}

    # Save CSVs
    for name in MODEL_NAMES:
        df_r = pd.DataFrame(records[name]).set_index("date")
        df_r.to_csv(f"output/backtest_{name.lower()}.csv")

    if as_json:
        monthly = {}
        for name in MODEL_NAMES:
            df_r = pd.DataFrame(records[name]).set_index("date")
            monthly[name] = [
                {
                    "date": d.strftime("%Y-%m"),
                    "strategy_pnl": float(df_r.loc[d, "strategy_pnl"]),
                    "bh_pnl": float(df_r.loc[d, "bh_pnl"]),
                    "alpha_pnl": float(df_r.loc[d, "alpha_pnl"]),
                    "pos_start": float(df_r.loc[d, "pos_start"]),
                    "pos_end": float(df_r.loc[d, "pos_end"]),
                    "logit": float(df_r.loc[d, "logit"]),
                    "signal": str(df_r.loc[d, "signal"]),
                }
                for d in df_r.index
            ]

        out = {
            "test_start": test_start,
            "generated_at": pd.Timestamp.today().isoformat(),
            "summary": summary,
            "year_by_year": yby,
            "monthly_pnl": monthly
        }
        print(json.dumps(out, indent=2))
        with open("output/backtest_result.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    else:
        W = 72
        print("=" * W)
        print(f"  BACKTEST RESULTS  |  V9.1  |  {test_start} -> latest")
        print("=" * W)
        print(f"  {'Model':<14} {'Sharpe':>7} {'NAV':>6} {'Alpha':>7} "
              f"{'MaxDD':>7} {'Calmar':>7} {'InMkt':>7}")
        print("-" * W)
        for name in MODEL_NAMES:
            m = summary[name]
            print(f"  {name:<14} {m['sharpe']:>7.3f} {m['nav']:>6.2f} "
                  f"{m['alpha_cum']:>+7.1f}% {m['max_dd']:>7.2f}% "
                  f"{m['calmar']:>7.3f} {m['in_market']:>6.1f}%")
        bh = summary[MODEL_NAMES[0]]
        print("-" * W)
        print(f"  {'BH SPX':<14} {bh['bh_sharpe']:>7.3f} {bh['bh_nav']:>6.2f}")
        print("=" * W)

        print("\n  Year-by-Year Strategy PnL:")
        all_years = sorted(yby[MODEL_NAMES[0]].keys())
        header    = f"  {'Year':<6}" + "".join(f"{n:>13}" for n in MODEL_NAMES)
        print(header); print("  " + "-"*(len(header)-2))
        for yr in all_years:
            row_str = f"  {yr:<6}"
            for name in MODEL_NAMES:
                v = yby[name].get(yr, 0.0)
                flag = " *" if abs(v) > 15 else "  "
                row_str += f"{v:>11.1f}%{flag}"
            print(row_str)
        print()
        print(f"  CSVs saved to output/backtest_*.csv")
        print("=" * W)

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    cfg      = load_config()
    as_json  = "--json" in sys.argv
    do_bt    = "--backtest" in sys.argv

    if not as_json:
        print("Fetching data...", flush=True)

    df_daily, macro = fetch_data(cfg["data_start"])
    df_m            = build_monthly(df_daily, macro)

    # Apply train_start filter for feature stability
    if cfg.get("train_start"):
        df_m = df_m[df_m.index >= cfg["train_start"]]

    if do_bt:
        run_backtest(df_m, cfg, as_json=as_json)
    else:
        run_live_signal(df_m, cfg, as_json=as_json)


if __name__ == "__main__":
    main()
