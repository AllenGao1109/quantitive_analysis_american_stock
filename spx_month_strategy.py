import warnings
warnings.filterwarnings("ignore")

import copy
import numpy as np
import pandas as pd
import pandas_datareader as pdr

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_OK = True
except ImportError:
    HMMLEARN_OK = False

try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False

# ============================================================
# CONFIG
# ============================================================
SEED            = 42
DATA_START      = "1990-01-01"
TRAIN_END       = "2015-12-31"
TEST_START      = "2016-01-01"
MIN_TRAIN       = 60
EPS             = 1e-12

DROP_PROXY_MID  = ["vix_regime", "vrp_high", "vrp_lag1", "vrp_lag3", "vix_post_spike"]
DROP_PROXY_BOM  = []

_SR_SHARED = dict(
    lambda_unc=0.20, lambda_hmm=0.15, lambda_hawkes=0.15,
    lambda_dd=0.25, lambda_tail=0.20, floor=0.25, shrink_mode="exp",
)

SOFT_CFGS_MID = {
    "LogReg_v92_SoftRisk_LO":    {**_SR_SHARED, "method": "prob_band", "base_scale": 1.2, "long_only": True},
    "LogReg_v92_SoftRisk_LS":    {**_SR_SHARED, "method": "prob_band", "base_scale": 1.2, "long_only": False},
    "ElasticNet_v92_SoftRisk_LO":{**_SR_SHARED, "method": "softsign",  "base_scale": 2.0, "long_only": True},
    "ElasticNet_v92_SoftRisk_LS":{**_SR_SHARED, "method": "softsign",  "base_scale": 2.0, "long_only": False},
    "Ensemble_v92_SoftRisk_LO":  {**_SR_SHARED, "method": "prob_band", "base_scale": 1.2, "long_only": True},
    "Ensemble_v92_SoftRisk_LS":  {**_SR_SHARED, "method": "prob_band", "base_scale": 1.2, "long_only": False},
}

SOFT_CFGS_BOM = {
    "LogReg_v92_BOM_SoftRisk_LO":     {**_SR_SHARED, "method": "prob_band", "base_scale": 1.2, "long_only": True},
    "LogReg_v92_BOM_SoftRisk_LS":     {**_SR_SHARED, "method": "prob_band", "base_scale": 1.2, "long_only": False},
    "ElasticNet_v92_BOM_SoftRisk_LO": {**_SR_SHARED, "method": "softsign",  "base_scale": 2.0, "long_only": True},
    "ElasticNet_v92_BOM_SoftRisk_LS": {**_SR_SHARED, "method": "softsign",  "base_scale": 2.0, "long_only": False},
    "Ensemble_v92_BOM_SoftRisk_LO":   {**_SR_SHARED, "method": "prob_band", "base_scale": 1.2, "long_only": True},
    "Ensemble_v92_BOM_SoftRisk_LS":   {**_SR_SHARED, "method": "prob_band", "base_scale": 1.2, "long_only": False},
}

# ============================================================
# HELPERS
# ============================================================
def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softsign_fn(x, scale=1.5):
    z = np.asarray(x, dtype=float) / max(float(scale), EPS)
    return z / (1.0 + np.abs(z))


def robust_rank01(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.Series(s).astype(float)
    return s.rank(pct=True).clip(0.0, 1.0)


def _safe(df, c):
    s = df[c]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s


def make_logit_position(logits, method="softsign", scale=1.5,
                        p_center=0.50, p_band=0.15, long_only=False):
    z = pd.Series(logits).fillna(0).astype(float)
    if method == "prob_band":
        z_sc = z / max(float(scale), EPS)
        p = pd.Series(sigmoid(z_sc.values), index=z.index)
        if long_only:
            pos = np.where(p >= p_center + p_band * 0.5, 1.0, 0.0)
        else:
            pos = np.where(p >= p_center + p_band * 0.5,  1.0,
                  np.where(p <= p_center - p_band * 0.5, -1.0, 0.0))
        pos = pd.Series(pos, index=z.index, dtype=float)
    else:
        pos = pd.Series(softsign_fn(z.values, scale=scale), index=z.index, dtype=float)
        if long_only:
            pos = pos.clip(lower=0.0)
    return pos.clip(-1.0, 1.0)


def annualized_sharpe(pnl_pct, periods=12):
    s = pd.Series(pnl_pct).dropna().astype(float)
    if len(s) < 3 or s.std() < EPS:
        return 0.0
    return float(np.sqrt(periods) * s.mean() / (s.std() + EPS))


def max_drawdown(pnl_pct):
    s = pd.Series(pnl_pct).fillna(0).astype(float) / 100.0
    nav = (1.0 + s).cumprod()
    return float((nav / nav.cummax() - 1.0).min() * 100.0)


def summarize_strategy_stats(pnl_pct, exposure=None, periods_per_year=12):
    pnl = pd.Series(pnl_pct).fillna(0).astype(float)
    nav = (1.0 + pnl / 100.0).cumprod()
    if exposure is None:
        exposure = pd.Series(index=pnl.index, data=np.nan)
    exposure = pd.Series(exposure).reindex(pnl.index)
    return dict(
        sharpe=round(annualized_sharpe(pnl, periods=periods_per_year), 3),
        final_nav=round(float(nav.iloc[-1]), 3) if len(nav) else 1.0,
        max_dd=round(max_drawdown(pnl), 2),
        avg_abs_position=round(float(exposure.abs().mean()), 3) if exposure.notna().any() else np.nan,
        n_periods=len(pnl.dropna()),
    )

# ============================================================
# DATA
# ============================================================
def load_data():
    print("Loading SPX...")
    data_end = (pd.Timestamp.today() + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

    raw = pdr.get_data_stooq("^SPX", start=DATA_START, end=data_end)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.columns = [c.strip().title() for c in raw.columns]
    df_daily = raw[["Open", "High", "Low", "Close"]].dropna().sort_index().copy()

    print("Loading VIX via FRED...")
    vix_raw = pdr.get_data_fred("VIXCLS", start=DATA_START, end=data_end)
    vix_raw.columns = ["vix"]
    vix_raw.index = pd.to_datetime(vix_raw.index).tz_localize(None)
    df_daily = df_daily.join(vix_raw, how="left")
    df_daily["vix"] = df_daily["vix"].ffill()

    df_daily["daily_return"]  = df_daily["Close"].pct_change() * 100
    df_daily["overnight_ret"] = (df_daily["Open"] / df_daily["Close"].shift(1) - 1) * 100
    df_daily["intraday_ret"]  = (df_daily["Close"] / df_daily["Open"] - 1) * 100
    df_daily["yearmonth"]     = df_daily.index.to_period("M")
    df_daily["day_rank"]      = df_daily.groupby("yearmonth").cumcount() + 1
    df_daily["day_rank_rev"]  = df_daily.groupby("yearmonth")["Close"].transform("count") - df_daily["day_rank"] + 1
    df_daily["high_252"]      = df_daily["Close"].rolling(252, min_periods=126).max().shift(1)
    df_daily["dist_52w"]      = (df_daily["Close"] - df_daily["high_252"]) / df_daily["high_252"] * 100

    print("Loading FRED macro...")
    tbl = pdr.get_data_fred("TB3MS",    start=DATA_START, end=data_end)
    lty = pdr.get_data_fred("GS10",     start=DATA_START, end=data_end)
    fed = pdr.get_data_fred("FEDFUNDS", start=DATA_START, end=data_end)
    baa = pdr.get_data_fred("BAA",      start=DATA_START, end=data_end)
    aaa = pdr.get_data_fred("AAA",      start=DATA_START, end=data_end)

    for d, n in [(tbl, "tbl"), (lty, "lty"), (fed, "fedfunds"), (baa, "baa"), (aaa, "aaa")]:
        d.columns = [n]
        d.index = pd.to_datetime(d.index).to_period("M").to_timestamp()

    dfy = (baa["baa"] - aaa["aaa"]).to_frame("dfy")
    dfr = (lty["lty"] - tbl["tbl"]).to_frame("dfr")
    macro = tbl.join([lty, fed, dfy, dfr], how="outer")
    macro["tbl_chg"]      = macro["tbl"].diff()
    macro["lty_chg"]      = macro["lty"].diff()
    macro["rate_rising"]  = (macro["fedfunds"].diff() > 0.1).astype(int)
    macro["rate_cutting"] = (macro["fedfunds"].diff() < -0.1).astype(int)
    macro["inverted"]     = (macro["dfr"] < 0).astype(int)
    return df_daily, macro

# ============================================================
# ADVANCED SIGNALS
# ============================================================
def kalman_local_level(y, q=1e-3, r=None, init_x=None, init_p=1.0):
    y = pd.Series(y).astype(float)
    if r is None:
        r = max(float(np.nanvar(y.dropna())), 1e-4)
    x = float(y.dropna().iloc[0] if init_x is None else init_x)
    p = float(init_p)
    level, err, var = [], [], []
    for val in y:
        x_pred, p_pred = x, p + q
        if pd.isna(val):
            level.append(x_pred); err.append(np.nan); var.append(p_pred)
        else:
            inn = val - x_pred
            s_  = p_pred + r
            k   = p_pred / s_
            x   = x_pred + k * inn
            p   = (1 - k) * p_pred
            level.append(x); err.append(inn); var.append(p)
    return (pd.Series(level, index=y.index),
            pd.Series(err,   index=y.index),
            pd.Series(var,   index=y.index))


def fit_hmm_regime(vix_series, n_states=3):
    s = pd.Series(vix_series).astype(float)
    feat = pd.DataFrame({"vix": s, "dvix": s.diff().fillna(0.0)}).dropna()
    regime_out = pd.Series(1, index=s.index, dtype=int)
    stress_prob_out = pd.Series(0.5, index=s.index, dtype=float)
    if HMMLEARN_OK and len(feat) >= 48:
        try:
            hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=SEED)
            hmm.fit(feat.values)
            labels = hmm.predict(feat.values)
            proba = hmm.predict_proba(feat.values)
            means_vix = [hmm.means_[i][0] for i in range(n_states)]
            stress_state = int(np.argmax(means_vix))
            calm_state = int(np.argmin(means_vix))
            regime_mapped = np.where(labels == stress_state, 2,
                            np.where(labels == calm_state, 0, 1))
            for ii, orig_idx in enumerate(feat.index):
                if orig_idx in regime_out.index:
                    regime_out.loc[orig_idx] = int(regime_mapped[ii])
                    stress_prob_out.loc[orig_idx] = float(proba[ii, stress_state])
        except Exception:
            pass
    else:
        q33, q66 = s.quantile(0.33), s.quantile(0.66)
        regime_out = pd.Series(np.where(s < q33, 0, np.where(s < q66, 1, 2)), index=s.index, dtype=int)
        stress_prob_out = pd.Series((s - s.min()) / (s.max() - s.min() + EPS), index=s.index, dtype=float)
    return regime_out.ffill().fillna(1).astype(int), stress_prob_out.ffill().fillna(0.5)


def hawkes_intensity(events, alpha=0.6, beta=0.75, base=None):
    e = pd.Series(events).fillna(0).astype(float).clip(lower=0)
    if base is None:
        base = max(0.01, float(e.mean()))
    lam = np.zeros(len(e), dtype=float)
    for i in range(len(e)):
        lam[i] = base + alpha * (float(e.iloc[i-1]) if i > 0 else 0.0) + beta * (lam[i-1] if i > 0 else base)
    return pd.Series(lam, index=e.index)


def rolling_ou_score(series, win=24):
    s = pd.Series(series).astype(float)
    mu = s.rolling(win, min_periods=win // 2).mean()
    sg = s.rolling(win, min_periods=win // 2).std()
    return ((s - mu) / (sg + EPS)).clip(-4, 4)


def garch_signals(monthly_ret, win=24):
    r = pd.Series(monthly_ret).astype(float)
    if ARCH_OK:
        try:
            am = arch_model(r.dropna(), vol="Garch", p=1, q=1, rescale=False)
            cond_vol = am.fit(disp="off").conditional_volatility.reindex(r.index)
        except Exception:
            cond_vol = r.ewm(span=win).std()
    else:
        cond_vol = r.ewm(span=win).std()
    return cond_vol.rename("garch_vol"), (r / (cond_vol + EPS)).clip(-4, 4).rename("garch_m_signal")


def dd_state_from_pnl(pnl_pct, halflife=6):
    s = pd.Series(pnl_pct).fillna(0).astype(float) / 100.0
    nav = (1.0 + s).cumprod()
    dd = nav / nav.cummax() - 1.0
    downside = (-s.clip(upper=0)).ewm(halflife=halflife, adjust=False).mean()
    dd_depth = (-dd).clip(lower=0)
    state = 0.6 * downside.rank(pct=True) + 0.4 * dd_depth.rank(pct=True)
    return state.clip(0, 1).rename("dd_state"), dd.rename("drawdown")


def build_risk_multiplier(df, floor=0.25, mode="exp",
                          lam_unc=0.20, lam_hmm=0.15, lam_hk=0.15,
                          lam_dd=0.25, lam_tail=0.20,
                          dd_state_col=None, tail_window=6, tail_return_col="remaining_ret"):
    def _rk(c):
        return robust_rank01(_safe(df, c)) if c in df.columns else pd.Series(0.0, index=df.index)

    unc = _rk("kalman_uncertainty")
    hmm = _rk("hmm_stress_prob")
    hk  = _rk("hawkes_vix_intensity")
    ddv = _rk(dd_state_col) if dd_state_col else pd.Series(0.0, index=df.index)

    if tail_return_col in df.columns:
        ret = _safe(df, tail_return_col).fillna(0).astype(float)
        tail_proxy = robust_rank01((-ret.clip(upper=0)).rolling(tail_window, min_periods=1).mean())
    else:
        tail_proxy = pd.Series(0.0, index=df.index)

    risk_score = lam_unc * unc + lam_hmm * hmm + lam_hk * hk + lam_dd * ddv + lam_tail * tail_proxy
    if mode == "exp":
        mult = float(floor) + (1.0 - float(floor)) * np.exp(-risk_score.astype(float))
    else:
        mult = float(floor) + (1.0 - float(floor)) / (1.0 + risk_score.astype(float))

    return pd.Series(mult, index=df.index).clip(lower=float(floor), upper=1.0), risk_score

# ============================================================
# MONTHLY FEATURE ENGINEERING
# - keep MID logic from original .py
# - replace EOM section with BOM-from-notebook logic
# ============================================================
def build_monthly(df_daily, macro):
    print("Building monthly features...")
    month_groups = {p: g.sort_index().copy() for p, g in df_daily.groupby("yearmonth")}
    sorted_periods = sorted(month_groups.keys())

    rows = []
    for i, period in enumerate(sorted_periods):
        g = month_groups[period]
        if len(g) < 5:
            continue

        close = g["Close"]
        dr = g["daily_return"]
        ts = period.to_timestamp()
        c0 = close.iloc[0]

        monthly_ret = (close.iloc[-1] / c0 - 1) * 100
        svar = (dr ** 2).sum()
        vix_avg = g["vix"].mean()
        rvm = dr.std() ** 2 * len(g)
        vrp = vix_avg ** 2 / 100 ** 2 * 12 - rvm / 10000
        overnight_avg = g["overnight_ret"].mean()
        intraday_avg = g["intraday_ret"].mean()
        on_minus_id = overnight_avg - intraday_avg
        overnight_vol = g["overnight_ret"].std()
        intraday_vol = g["intraday_ret"].std()
        dist_52w_end = g["dist_52w"].iloc[-1]

        d13 = g[g["day_rank"] <= 3]
        tom_early = ((d13["Close"].iloc[-1] / d13["Close"].iloc[0] - 1) * 100 if len(d13) >= 2 else np.nan)

        d_last = g[g["day_rank_rev"] <= 1]
        tom_last = float(d_last["daily_return"].iloc[0]) if len(d_last) >= 1 else np.nan
        vix_eom = float(d_last["vix"].iloc[0]) if len(d_last) >= 1 else np.nan

        up_ratio_month = float((dr > 0).mean())
        on_ret_month = g["overnight_ret"].mean()
        id_ret_month = g["intraday_ret"].mean()
        on_minus_id_month = on_ret_month - id_ret_month
        overnight_vol_month = g["overnight_ret"].std()
        intraday_vol_month = g["intraday_ret"].std()

        # MID features / target from original .py
        mid = g[(g["day_rank"] >= 7) & (g["day_rank"] <= 12)]
        vix_mid = mid["vix"].mean() if len(mid) > 0 else np.nan

        d8_12 = g[(g["day_rank"] >= 8) & (g["day_rank"] <= 12)]
        if len(d8_12) > 0:
            c_mid = d8_12["Close"].mean()
            vix_day10 = d8_12["vix"].mean()
            som_ret = (c_mid / c0 - 1) * 100
            first_mid = g[g["day_rank"] <= d8_12["day_rank"].max()]
            rv_d10 = first_mid["daily_return"].std() ** 2 * len(first_mid)
            vrp_d10 = vix_day10 ** 2 / 100 ** 2 * 12 - rv_d10 / 10000
            vol_d10 = first_mid["daily_return"].std()
            up_ratio_d10 = float((first_mid["daily_return"] > 0).mean())
            on_ret_d10 = first_mid["overnight_ret"].mean()
            id_ret_d10 = first_mid["intraday_ret"].mean()
            mom_d10 = (c_mid / c0 - 1) * 100
            neg_rets = first_mid["daily_return"][first_mid["daily_return"] < 0]
            dvp = float(np.sqrt((neg_rets ** 2).mean())) if len(neg_rets) > 0 else 0.0
        else:
            c_mid = vix_day10 = som_ret = vrp_d10 = vol_d10 = np.nan
            up_ratio_d10 = on_ret_d10 = id_ret_d10 = mom_d10 = dvp = np.nan

        d_eom_this = g[g["day_rank_rev"] <= 2]["Close"]
        if i + 1 < len(sorted_periods):
            g_next = month_groups[sorted_periods[i + 1]]
            d_eom_next = g_next[g_next["day_rank"] <= 2]["Close"]
        else:
            d_eom_next = pd.Series(dtype=float)
        eom_closes = pd.concat([d_eom_this, d_eom_next])
        c_eom = eom_closes.mean() if len(eom_closes) > 0 else close.iloc[-1]
        remaining_ret = (c_eom / c_mid - 1) * 100 if not pd.isna(c_mid) else np.nan

        # BOM features / target from notebook
        som_window_bom = g[(g["day_rank"] >= 1) & (g["day_rank"] <= 8)]
        if len(som_window_bom) >= 2:
            c_som_bom = som_window_bom["Close"].mean()
            som_ret_bom = (c_som_bom / c0 - 1) * 100
        else:
            som_ret_bom = np.nan

        if i >= 1:
            g_prev = month_groups[sorted_periods[i - 1]]
            prev_last5 = g_prev.tail(5).copy()

            prev_last5_ret = (prev_last5["Close"].iloc[-1] / prev_last5["Close"].iloc[0] - 1) * 100 if len(prev_last5) >= 2 else np.nan
            prev_last5_vol = prev_last5["daily_return"].std()
            prev_last5_up_ratio = float((prev_last5["daily_return"] > 0).mean()) if len(prev_last5) > 0 else np.nan
            prev_last5_on_ret = prev_last5["overnight_ret"].mean() if len(prev_last5) > 0 else np.nan
            prev_last5_id_ret = prev_last5["intraday_ret"].mean() if len(prev_last5) > 0 else np.nan
            prev_last5_on_minus_id = prev_last5_on_ret - prev_last5_id_ret if pd.notna(prev_last5_on_ret) and pd.notna(prev_last5_id_ret) else np.nan
            prev_last5_on_vol = prev_last5["overnight_ret"].std()
            prev_last5_id_vol = prev_last5["intraday_ret"].std()
            prev_last5_vix_avg = prev_last5["vix"].mean() if len(prev_last5) > 0 else np.nan
            prev_last5_vix_chg = prev_last5["vix"].iloc[-1] - prev_last5["vix"].iloc[0] if len(prev_last5) >= 2 else np.nan
            prev_last5_rv = prev_last5["daily_return"].std() ** 2 * len(prev_last5) if len(prev_last5) > 1 else np.nan
            prev_last5_vrp = (prev_last5_vix_avg ** 2 / 100 ** 2 * 12 - prev_last5_rv / 10000) if pd.notna(prev_last5_vix_avg) and pd.notna(prev_last5_rv) else np.nan
            prev_last5_vix_last = float(prev_last5["vix"].iloc[-1]) if len(prev_last5) > 0 else np.nan
            prev_last5_close_to_52w = float(prev_last5["dist_52w"].iloc[-1]) if len(prev_last5) > 0 else np.nan
        else:
            prev_last5_ret = prev_last5_vol = prev_last5_up_ratio = np.nan
            prev_last5_on_ret = prev_last5_id_ret = prev_last5_on_minus_id = np.nan
            prev_last5_on_vol = prev_last5_id_vol = prev_last5_vix_avg = np.nan
            prev_last5_vix_chg = prev_last5_vrp = prev_last5_vix_last = np.nan
            prev_last5_close_to_52w = np.nan

        rows.append(dict(
            ts=ts,
            monthly_return=monthly_ret,
            svar=svar,
            vix_month=vix_avg,
            vix_mid=vix_mid,
            vix_day10=vix_day10,
            vrp=vrp,
            vrp_d10=vrp_d10,
            overnight_avg=overnight_avg,
            intraday_avg=intraday_avg,
            on_minus_id=on_minus_id,
            overnight_vol=overnight_vol,
            intraday_vol=intraday_vol,
            dist_52w=dist_52w_end,
            tom_early=tom_early,
            tom_last=tom_last,
            som_ret=som_ret,
            remaining_ret=remaining_ret,
            dvp=dvp,
            mom_d10=mom_d10,
            vol_d10=vol_d10,
            up_ratio_d10=up_ratio_d10,
            on_ret_d10=on_ret_d10,
            id_ret_d10=id_ret_d10,
            vix_eom=vix_eom,
            up_ratio_month=up_ratio_month,
            on_ret_month=on_ret_month,
            id_ret_month=id_ret_month,
            on_minus_id_month=on_minus_id_month,
            overnight_vol_month=overnight_vol_month,
            intraday_vol_month=intraday_vol_month,
            som_ret_bom=som_ret_bom,
            prev_last5_ret=prev_last5_ret,
            prev_last5_vol=prev_last5_vol,
            prev_last5_up_ratio=prev_last5_up_ratio,
            prev_last5_on_ret=prev_last5_on_ret,
            prev_last5_id_ret=prev_last5_id_ret,
            prev_last5_on_minus_id=prev_last5_on_minus_id,
            prev_last5_on_vol=prev_last5_on_vol,
            prev_last5_id_vol=prev_last5_id_vol,
            prev_last5_vix_avg=prev_last5_vix_avg,
            prev_last5_vix_chg=prev_last5_vix_chg,
            prev_last5_vrp=prev_last5_vrp,
            prev_last5_vix_last=prev_last5_vix_last,
            prev_last5_close_to_52w=prev_last5_close_to_52w,
        ))

    df_m = pd.DataFrame(rows).set_index("ts")
    df_m.index = pd.to_datetime(df_m.index)

    df_m["rev1m"] = df_m["monthly_return"].shift(1)
    df_m["mom3m"] = df_m["monthly_return"].rolling(3).sum().shift(1)
    df_m["mom6m"] = df_m["monthly_return"].rolling(6).sum().shift(2)
    df_m["mom12m_skip1"] = df_m["monthly_return"].rolling(12).sum().shift(2)
    df_m["mom24m"] = df_m["monthly_return"].rolling(24).sum().shift(2)
    df_m["vol6m"] = df_m["monthly_return"].rolling(6).std().shift(1)
    df_m["vol12m"] = df_m["monthly_return"].rolling(12).std().shift(1)
    df_m["svar_lag1"] = df_m["svar"].shift(1)
    df_m["vrp_lag1"] = df_m["vrp"].shift(1)
    df_m["vrp_lag3"] = df_m["vrp"].rolling(3).mean().shift(1)
    df_m["vix_zscore12"] = (df_m["vix_month"] - df_m["vix_month"].rolling(12).mean()) / (df_m["vix_month"].rolling(12).std() + EPS)
    df_m["vix_chg1m"] = df_m["vix_month"].diff(1)
    df_m["vix_chg3m"] = df_m["vix_month"].diff(3)
    df_m["vix_regime"] = df_m["vix_month"].apply(lambda v: 0 if v < 15 else 1 if v < 20 else 2 if v < 30 else 3)
    df_m["high_vix"] = (df_m["vix_month"] > 25).astype(int)
    df_m["vix_spike"] = (df_m["vix_chg1m"] > 5).astype(int)
    df_m["vix_post_spike"] = df_m["vix_spike"].shift(1).fillna(0).astype(int)
    df_m["vrp_positive"] = (df_m["vrp"] > 0).astype(int)
    df_m["vrp_high"] = (df_m["vrp"] > df_m["vrp"].rolling(12).quantile(0.75)).astype(int)
    df_m["dvp_lag1"] = df_m["dvp"].shift(1)
    df_m["vrp_x_vol12"] = df_m["vrp"] * df_m["vol12m"].shift(1)
    df_m["tsmom"] = df_m["mom12m_skip1"] / (df_m["vol12m"].shift(1) + EPS)
    for moy in [1, 9, 10, 12]:
        df_m[f"month_{moy}"] = (df_m.index.month == moy).astype(int)

    macro_m = macro.copy()
    macro_m.index = pd.to_datetime(macro_m.index)
    macro_cols = ["tbl", "lty", "fedfunds", "dfy", "dfr",
                  "tbl_chg", "lty_chg", "rate_rising", "rate_cutting", "inverted"]
    df_m = df_m.join(macro_m[macro_cols], how="left")
    for c in macro_cols:
        if c in df_m.columns:
            df_m[c] = df_m[c].ffill()

    df_m["macro_stress"] = (
        (df_m["vix_zscore12"] > 1).astype(int)
        * df_m["inverted"].fillna(0).astype(int)
        * (df_m["lty_chg"] > 0.2).fillna(0).astype(int)
    )
    df_m["stress_roll3"] = df_m["macro_stress"].rolling(3, min_periods=1).mean()
    df_m["vrp_x_vix"] = df_m["vrp"] * df_m["vix_regime"]
    df_m["mom_x_stress"] = df_m["mom12m_skip1"] * df_m["macro_stress"]
    df_m["dist_x_vix"] = df_m["dist_52w"] * df_m["vix_regime"]

    df_m["y_mid"] = (df_m["remaining_ret"] > 0).astype(float)
    df_m["y_bom"] = (df_m["som_ret_bom"] > 0).astype(float)
    return df_m


def add_v92_features(df_m):
    print("Adding v9.2 advanced features...")
    df = df_m.copy()
    kf_level, _, kf_var = kalman_local_level(df["monthly_return"], q=0.05)
    kf_som, _, _ = kalman_local_level(df["som_ret"], q=0.03)
    df["kalman_trend_ret"] = kf_level.shift(1)
    df["kalman_trend_som"] = kf_som.shift(1)
    df["kalman_uncertainty"] = np.sqrt(kf_var.shift(1).clip(lower=0))
    df["kalman_signal"] = df["kalman_trend_ret"] / (df["kalman_uncertainty"] + EPS)
    df["hmm_regime"], df["hmm_stress_prob"] = fit_hmm_regime(df["vix_month"], n_states=3)
    df["ou_z_vrp"] = rolling_ou_score(df["vrp"], win=24)
    df["ou_z_vrp_d10"] = rolling_ou_score(df["vrp_d10"].fillna(df["vrp"]), win=24)
    df["ou_z_prev5"] = rolling_ou_score(df["prev_last5_vrp"].fillna(df["vrp"]), win=24)
    df["ou_z_vix"] = rolling_ou_score(df["vix_month"], win=24)
    df["ou_mr_signal"] = (-df["ou_z_vrp"]).clip(-4, 4)
    q80 = df["vix_month"].diff().rolling(24).quantile(0.80)
    vix_jump = (df["vix_month"].diff() > q80).astype(float)
    df["hawkes_vix_intensity"] = hawkes_intensity(vix_jump, alpha=0.8, beta=0.55)
    df["hawkes_jump_flag"] = (vix_jump > 0).astype(int)
    gvol, gsig = garch_signals(df["monthly_return"], win=24)
    df["garch_vol"] = gvol
    df["garch_m_signal"] = gsig
    df["kalman_x_hmm"] = df["kalman_signal"] * df["hmm_stress_prob"]
    df["ou_x_hmm"] = df["ou_mr_signal"] * df["hmm_stress_prob"]
    df["hawkes_x_vix"] = df["hawkes_vix_intensity"] * df["vix_zscore12"]
    df["garch_x_stress"] = df["garch_m_signal"] * df["macro_stress"]
    return df

# ============================================================
# FEATURE SETS
# ============================================================
V92_ADV_FEATURES = [
    "hmm_regime", "hmm_stress_prob",
    "kalman_trend_ret", "kalman_trend_som", "kalman_uncertainty", "kalman_signal",
    "ou_z_vrp", "ou_z_vrp_d10", "ou_z_prev5", "ou_z_vix", "ou_mr_signal",
    "hawkes_vix_intensity", "hawkes_jump_flag",
    "garch_vol", "garch_m_signal",
    "kalman_x_hmm", "ou_x_hmm", "hawkes_x_vix", "garch_x_stress",
]

ALL_MID_BASE_FEATURES = [
    "vrp", "vrp_d10", "vrp_lag1", "vrp_lag3", "vrp_positive", "vrp_high",
    "svar", "svar_lag1",
    "vix_day10", "vix_mid", "vix_zscore12", "vix_chg1m", "vix_chg3m",
    "vix_regime", "high_vix", "vix_spike", "vix_post_spike",
    "rev1m", "mom3m", "mom6m", "mom12m_skip1", "mom24m", "vol6m", "vol12m",
    "dist_52w", "mom_d10", "vol_d10", "up_ratio_d10", "on_ret_d10", "id_ret_d10",
    "overnight_avg", "intraday_avg", "on_minus_id", "overnight_vol", "intraday_vol",
    "tom_early", "tom_last",
    "dfy", "dfr", "tbl", "lty", "tbl_chg", "lty_chg",
    "rate_rising", "rate_cutting", "inverted",
    "macro_stress", "stress_roll3", "vrp_x_vix", "mom_x_stress", "dist_x_vix",
    "month_1", "month_9", "month_10", "month_12",
    "vrp_x_vol12", "tsmom", "dvp_lag1",
]

ALL_BOM_BASE_FEATURES = [
    "rev1m", "mom3m", "mom6m", "mom12m_skip1", "mom24m",
    "vol6m", "vol12m", "dist_52w",
    "vrp_lag1", "vrp_lag3", "vrp_positive", "vrp_high",
    "svar_lag1",
    "vix_zscore12", "vix_chg1m", "vix_chg3m", "vix_regime",
    "high_vix", "vix_spike", "vix_post_spike",
    "dfy", "dfr", "tbl", "lty", "tbl_chg", "lty_chg",
    "rate_rising", "rate_cutting", "inverted",
    "macro_stress", "stress_roll3", "vrp_x_vix", "mom_x_stress", "dist_x_vix",
    "month_1", "month_9", "month_10", "month_12",
    "vrp_x_vol12", "tsmom",
    "prev_last5_ret", "prev_last5_vol", "prev_last5_up_ratio",
    "prev_last5_on_ret", "prev_last5_id_ret", "prev_last5_on_minus_id",
    "prev_last5_on_vol", "prev_last5_id_vol",
    "prev_last5_vix_avg", "prev_last5_vix_chg", "prev_last5_vrp",
    "prev_last5_vix_last", "prev_last5_close_to_52w",
]


def get_mid_features(df_v92):
    base = [c for c in ALL_MID_BASE_FEATURES if c in df_v92.columns and c not in DROP_PROXY_MID]
    adv = [c for c in V92_ADV_FEATURES if c in df_v92.columns]
    feats = list(dict.fromkeys(base + adv))
    print(f"  MID features: {len(feats)} total ({len(base)} base + {len(adv)} advanced)")
    return feats


def get_bom_features(df_v92):
    base = [c for c in ALL_BOM_BASE_FEATURES if c in df_v92.columns and c not in DROP_PROXY_BOM]
    adv = [c for c in V92_ADV_FEATURES if c in df_v92.columns]
    feats = list(dict.fromkeys(base + adv))
    print(f"  BOM features: {len(feats)} total ({len(base)} base + {len(adv)} advanced)")
    return feats

# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward_generic(df_v92, features, target_col, model_prefix):
    print(f"Running walk-forward: {model_prefix} target={target_col}...")
    state_cols = [
        "kalman_uncertainty", "hmm_stress_prob", "hawkes_vix_intensity",
        "som_ret", "som_ret_bom", "remaining_ret", "monthly_return", "vix_month", "vrp",
        "vix_spike", "vix_regime"
    ]
    needed = list(dict.fromkeys(features + [target_col] + [c for c in state_cols if c in df_v92.columns]))
    df_feat = df_v92[[c for c in needed if c in df_v92.columns]].dropna(subset=features + [target_col]).copy()

    idx = df_feat.index
    logits = {f"LogReg_{model_prefix}": [], f"ElasticNet_{model_prefix}": []}
    idx_out = []

    logreg_tmpl = LogisticRegression(max_iter=3000, C=0.1, random_state=SEED)
    enet_tmpl = SGDClassifier(loss="log_loss", penalty="elasticnet",
                              alpha=0.005, l1_ratio=0.5, max_iter=3000,
                              random_state=SEED, tol=1e-4)

    for t in range(MIN_TRAIN, len(idx)):
        date = idx[t]
        idx_out.append(date)
        df_tr = df_feat[df_feat.index < date]
        df_te = df_feat[df_feat.index == date]

        if len(df_tr) < MIN_TRAIN or len(df_te) != 1:
            logits[f"LogReg_{model_prefix}"].append(0.0)
            logits[f"ElasticNet_{model_prefix}"].append(0.0)
            continue

        X_tr = df_tr[features].fillna(0).values
        y_tr = df_tr[target_col].values
        X_te = df_te[features].fillna(0).values

        if len(np.unique(y_tr)) < 2:
            logits[f"LogReg_{model_prefix}"].append(0.0)
            logits[f"ElasticNet_{model_prefix}"].append(0.0)
            continue

        sc = StandardScaler().fit(X_tr)
        Xf = sc.transform(X_tr)
        Xt = sc.transform(X_te)

        m1 = clone(logreg_tmpl)
        m1.fit(Xf, y_tr)
        logits[f"LogReg_{model_prefix}"].append(float(m1.decision_function(Xt)[0]))

        m2 = clone(enet_tmpl)
        m2.fit(Xf, y_tr)
        logits[f"ElasticNet_{model_prefix}"].append(float(m2.decision_function(Xt)[0]))

    df_wf = pd.DataFrame(logits, index=idx_out)
    state_avail = [c for c in state_cols if c in df_feat.columns]
    df_wf = df_wf.join(df_feat[state_avail], how="left")
    df_wf = df_wf.loc[:, ~df_wf.columns.duplicated(keep="first")]
    df_wf[f"Ensemble_{model_prefix}"] = df_wf[[f"LogReg_{model_prefix}", f"ElasticNet_{model_prefix}"]].mean(axis=1)
    return df_wf

# ============================================================
# POSITION BUILDERS
# ============================================================
def build_base_position_history(signal_hist, base_scale=1.5):
    out = {}
    for sig in signal_hist.columns:
        ser = _safe(signal_hist, sig)
        pos_ls = make_logit_position(ser, method="softsign", scale=base_scale, long_only=False)
        pos_lo = pos_ls.clip(lower=0.0, upper=1.0)
        out[f"{sig}_Base_LS"] = pos_ls
        out[f"{sig}_Base_LO"] = pos_lo
    return pd.DataFrame(out, index=signal_hist.index)


def build_softrisk_position_history(signal_hist, df_v92, soft_cfg_dict, tail_return_col, pnl_return_col):
    df_full = signal_hist.copy()
    state_cols = [c for c in [
        "kalman_uncertainty", "hmm_stress_prob", "hawkes_vix_intensity",
        "monthly_return", "remaining_ret", "som_ret", "som_ret_bom", "vix_month", "vrp"
    ] if c in df_v92.columns]
    df_full = df_full.join(df_v92[state_cols], how="left")

    sigs = list(signal_hist.columns)
    for sig in sigs:
        naive_pos = make_logit_position(_safe(df_full, sig), method="softsign", scale=1.5)
        naive_pnl = naive_pos * _safe(df_full, pnl_return_col).fillna(0)
        dd_state, _ = dd_state_from_pnl(naive_pnl, halflife=6)
        df_full[f"{sig}_dd_state"] = dd_state.values

    out = {}
    for key, cfg in soft_cfg_dict.items():
        sig = key.replace("_SoftRisk_LS", "").replace("_SoftRisk_LO", "")
        if sig not in df_full.columns:
            continue
        dd_col = f"{sig}_dd_state" if f"{sig}_dd_state" in df_full.columns else None
        mult, _ = build_risk_multiplier(
            df_full,
            floor=cfg["floor"],
            mode=cfg["shrink_mode"],
            lam_unc=cfg["lambda_unc"],
            lam_hmm=cfg["lambda_hmm"],
            lam_hk=cfg["lambda_hawkes"],
            lam_dd=cfg["lambda_dd"],
            lam_tail=cfg["lambda_tail"],
            dd_state_col=dd_col,
            tail_return_col=tail_return_col,
        )
        base_pos = make_logit_position(
            _safe(df_full, sig),
            method=cfg["method"],
            scale=cfg["base_scale"],
            long_only=cfg["long_only"],
        )
        risk_pos = (base_pos * mult).clip(lower=0.0 if cfg["long_only"] else -1.0, upper=1.0)
        out[key] = risk_pos
    return pd.DataFrame(out, index=signal_hist.index) if out else pd.DataFrame(index=signal_hist.index)


def build_logitstrength_position_history(signal_hist):
    out = {}
    for sig in signal_hist.columns:
        ser = _safe(signal_hist, sig).astype(float)
        pct = ser.expanding().rank(pct=True)
        raw = 2.0 * pct - 1.0
        out[f"{sig}_LogitStr_LS"] = raw.clip(-1.0, 1.0)
        out[f"{sig}_LogitStr_LO"] = raw.clip(0.0, 1.0)
    return pd.DataFrame(out, index=signal_hist.index)

# ============================================================
# PNL
# ============================================================
def compute_mid_pnl(pos_mid, df_v92, oos_start=TEST_START):
    pos = pos_mid.reindex(df_v92.index).fillna(0)
    pnl = (pos * _safe(df_v92, "remaining_ret").fillna(0)).rename("strategy_pnl")
    df_out = pd.DataFrame({
        "strategy_pnl": pnl,
        "bh_pnl": _safe(df_v92, "remaining_ret").fillna(0),
        "alpha_pnl": pnl - _safe(df_v92, "remaining_ret").fillna(0),
        "position": pos,
        "som_ret": _safe(df_v92, "som_ret").fillna(0),
        "som_ret_bom": _safe(df_v92, "som_ret_bom").fillna(0),
        "remaining_ret": _safe(df_v92, "remaining_ret").fillna(0),
    })
    return df_out[df_out.index >= oos_start]


def compute_bom_pnl(pos_bom, df_v92, oos_start=TEST_START):
    pos = pos_bom.reindex(df_v92.index).fillna(0)
    bom_ret = _safe(df_v92, "som_ret_bom").fillna(0)
    pnl = (pos * bom_ret).rename("strategy_pnl")
    df_out = pd.DataFrame({
        "strategy_pnl": pnl,
        "bh_pnl": bom_ret,
        "alpha_pnl": pnl - bom_ret,
        "position": pos,
        "som_ret_bom": bom_ret,
        "remaining_ret": _safe(df_v92, "remaining_ret").fillna(0),
    })
    return df_out[df_out.index >= oos_start]


def compute_combined_pnl(pos_mid, pos_bom, df_v92, oos_start=TEST_START):
    pm = pos_mid.reindex(df_v92.index).fillna(0)
    pb = pos_bom.reindex(df_v92.index).fillna(0)
    bom_ret = _safe(df_v92, "som_ret_bom").fillna(0)
    rem = _safe(df_v92, "remaining_ret").fillna(0)
    bh = _safe(df_v92, "monthly_return").fillna(0)
    pnl = (pb * bom_ret + pm * rem).rename("strategy_pnl")
    df_out = pd.DataFrame({
        "strategy_pnl": pnl,
        "bh_pnl": bh,
        "alpha_pnl": pnl - bh,
        "position": (pb.abs() + pm.abs()) / 2.0,
        "pos_bom": pb,
        "pos_mid": pm,
        "som_ret_bom": bom_ret,
        "remaining_ret": rem,
    })
    return df_out[df_out.index >= oos_start]

# ============================================================
# REPORTING
# ============================================================
def summarize_position_family(position_hist_df, df_v92, segment="mid"):
    rows = []
    for col in position_hist_df.columns:
        pos = _safe(position_hist_df, col)
        if segment == "mid":
            pnl_df = compute_mid_pnl(pos, df_v92)
        elif segment == "bom":
            pnl_df = compute_bom_pnl(pos, df_v92)
        else:
            raise ValueError("segment must be 'mid' or 'bom'")
        stats = summarize_strategy_stats(pnl_df["strategy_pnl"], exposure=pnl_df["position"])
        rows.append({"model": col, **stats})
    return pd.DataFrame(rows).sort_values(["sharpe", "final_nav"], ascending=False).reset_index(drop=True)


def pick_top_ls_lo(rank_df, n_ls=2, n_lo=2):
    rank_df = rank_df.copy()
    ls_df = rank_df[rank_df["model"].str.endswith("_LS")].head(n_ls)
    lo_df = rank_df[rank_df["model"].str.endswith("_LO")].head(n_lo)
    out = pd.concat([ls_df, lo_df], axis=0).reset_index(drop=True)
    return out


def build_cross_results(bom_top_df, mid_top_df, bom_hist, mid_hist, df_v92):
    rows = []
    for bom_model in bom_top_df["model"].tolist():
        for mid_model in mid_top_df["model"].tolist():
            df_pnl = compute_combined_pnl(_safe(mid_hist, mid_model), _safe(bom_hist, bom_model), df_v92)
            stats = summarize_strategy_stats(df_pnl["strategy_pnl"], exposure=df_pnl["position"])
            rows.append({
                "bom_model": bom_model,
                "mid_model": mid_model,
                "combined_name": f"BOM[{bom_model}]__MID[{mid_model}]",
                **stats,
                "avg_bom_abs_position": round(float(_safe(bom_hist, bom_model).loc[_safe(bom_hist, bom_model).index >= TEST_START].abs().mean()), 3),
                "avg_mid_abs_position": round(float(_safe(mid_hist, mid_model).loc[_safe(mid_hist, mid_model).index >= TEST_START].abs().mean()), 3),
            })
    return pd.DataFrame(rows).sort_values(["sharpe", "final_nav"], ascending=False).reset_index(drop=True)





def build_position_action_table(position_history_df):
    rows = []
    for col in position_history_df.columns:
        ser = _safe(position_history_df, col).dropna()
        if len(ser) == 0:
            current_pos, after_pos = np.nan, np.nan
        elif len(ser) == 1:
            current_pos, after_pos = 0.0, float(ser.iloc[-1])
        else:
            current_pos, after_pos = float(ser.iloc[-2]), float(ser.iloc[-1])
        delta = after_pos - current_pos if (pd.notna(current_pos) and pd.notna(after_pos)) else np.nan
        if pd.isna(delta):
            action = "NA"
        elif abs(delta) < 1e-8:
            action = "Hold"
        elif delta > 0:
            action = f"Increase by {delta:.3f}"
        else:
            action = f"Decrease by {abs(delta):.3f}"
        rows.append(dict(model=col, current_position=current_pos,
                         recommended_position_change=delta,
                         after_position=after_pos, action=action))
    return pd.DataFrame(rows)



def get_active_segment(df_daily, split_day=12):
    df = df_daily.copy().sort_index()
    latest_date = df.index.max()
    latest_month = latest_date.to_period("M")
    df_this_month = df[df.index.to_period("M") == latest_month]

    if len(df_this_month) == 0:
        return "BOM", np.nan, latest_date

    current_day_rank = int(df_this_month["day_rank"].iloc[-1])

    if current_day_rank <= split_day:
        active_segment = "BOM"
    else:
        active_segment = "MID"

    return active_segment, current_day_rank, latest_date


def build_current_df(df_daily, df_wf_bom, df_wf_mid, bom_hist, mid_hist, df_v92,
                     split_day=12, oos_start=TEST_START):
    rows = []

    active_segment, current_day_rank, latest_date = get_active_segment(df_daily, split_day=split_day)

    latest_bom = df_wf_bom.index.max()
    latest_mid = df_wf_mid.index.max()

    if active_segment == "BOM":
        source_hist = bom_hist
        source_wf = df_wf_bom
        latest_idx = latest_bom
        seg_name = "BOM"
    else:
        source_hist = mid_hist
        source_wf = df_wf_mid
        latest_idx = latest_mid
        seg_name = "MID"

    for col in source_hist.columns:
        sig = col.split("_Base_")[0].split("_SoftRisk_")[0].split("_LogitStr_")[0]

        if "_SoftRisk_" in col:
            family = "SoftRisk"
        elif "_Base_" in col:
            family = "Base"
        elif "_LogitStr_" in col:
            family = "LogitStrength"
        else:
            family = "Unknown"

        latest_logit = float(_safe(source_wf, sig).loc[latest_idx]) if sig in source_wf.columns else np.nan

        ser = _safe(source_hist, col).dropna()
        target_position = float(ser.iloc[-1]) if len(ser) else np.nan

        # OOS stats
        if seg_name == "BOM":
            pnl_df = compute_bom_pnl(_safe(source_hist, col), df_v92, oos_start=oos_start)
        else:
            pnl_df = compute_mid_pnl(_safe(source_hist, col), df_v92, oos_start=oos_start)

        stats = summarize_strategy_stats(
            pnl_df["strategy_pnl"],
            exposure=pnl_df["position"]
        )

        rows.append(dict(
            segment=seg_name,
            current_day_rank=current_day_rank,
            asof_date=latest_date,
            model=col,
            family=family,
            latest_logit=round(latest_logit, 4) if pd.notna(latest_logit) else np.nan,
            latest_prob_up=round(float(sigmoid(latest_logit)), 4) if pd.notna(latest_logit) else np.nan,
            target_position=round(target_position, 4) if pd.notna(target_position) else np.nan,
            sharpe_test=stats["sharpe"],
            final_nav_test=stats["final_nav"],
        ))

    return (pd.DataFrame(rows)
            .sort_values(["sharpe_test", "final_nav_test"], ascending=False)
            .reset_index(drop=True))[[
                "segment",
                "current_day_rank",
                "asof_date",
                "model",
                "family",
                "latest_logit",
                "latest_prob_up",
                "target_position",
                "sharpe_test",
                "final_nav_test",
            ]]



def build_backtest_df(bom_hist, mid_hist, df_v92, best_bom_model, best_mid_model, df_cross=None):
    parts = []
    for col in bom_hist.columns:
        df_pnl = compute_bom_pnl(_safe(bom_hist, col), df_v92)
        df_pnl["model_variant"] = col
        df_pnl["segment"] = "BOM"
        parts.append(df_pnl)
    for col in mid_hist.columns:
        df_pnl = compute_mid_pnl(_safe(mid_hist, col), df_v92)
        df_pnl["model_variant"] = col
        df_pnl["segment"] = "MID"
        parts.append(df_pnl)

    df_best = compute_combined_pnl(_safe(mid_hist, best_mid_model), _safe(bom_hist, best_bom_model), df_v92)
    df_best["model_variant"] = f"CombinedBest__BOM[{best_bom_model}]__MID[{best_mid_model}]"
    df_best["segment"] = "COMBINED_BEST"
    parts.append(df_best)

    if df_cross is not None and len(df_cross) > 0:
        for _, row in df_cross.iterrows():
            df_pair = compute_combined_pnl(_safe(mid_hist, row["mid_model"]), _safe(bom_hist, row["bom_model"]), df_v92)
            df_pair["model_variant"] = row["combined_name"]
            df_pair["segment"] = "COMBINED_TOP4_CROSS"
            parts.append(df_pair)

    backtest_df = pd.concat(parts).sort_index()
    backtest_df.index.name = "month"
    return backtest_df

# ============================================================
# MAIN
# ============================================================
def main():
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

    backtest_df = build_backtest_df(bom_hist, mid_hist, df_v92, best_bom_model, best_mid_model, df_cross=df_cross)
    current_df = build_current_df(
        df_daily=df_daily,
        df_wf_bom=df_wf_bom,
        df_wf_mid=df_wf_mid,
        bom_hist=bom_hist,
        mid_hist=mid_hist,
        df_v92=df_v92,
        split_day=12,
        oos_start=TEST_START,
    )

    print("=== BEST BOM MODEL ===")
    print(best_bom_model)
    print(bom_rank.head(10).to_string(index=False))

    print("=== BEST MID MODEL ===")
    print(best_mid_model)
    print(mid_rank.head(10).to_string(index=False))

    print("=== BOM TOP4 (2 LS + 2 LO) ===")
    print(bom_top4.to_string(index=False))

    print("=== MID TOP4 (2 LS + 2 LO) ===")
    print(mid_top4.to_string(index=False))

    print("=== TOP4 x TOP4 CROSS RESULTS ===")
    print(df_cross.drop(columns=["combined_name"], errors="ignore").to_string(index=False))

    combined_best_df = compute_combined_pnl(_safe(mid_hist, best_mid_model), _safe(bom_hist, best_bom_model), df_v92)
    combined_stats = summarize_strategy_stats(combined_best_df["strategy_pnl"], exposure=combined_best_df["position"])
    print("=== COMBINED BEST SUMMARY (overall best BOM + overall best MID) ===")
    print(combined_stats)

    cross_best = df_cross.iloc[0].to_dict() if len(df_cross) else {}
    print("=== COMBINED TOP4 CROSS BEST ===")
    print(cross_best)

    print("=== CURRENT DECISION TABLE ===")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(current_df.to_string(index=False))

    return backtest_df, current_df, bom_rank, mid_rank, bom_top4, mid_top4, df_cross


if __name__ == "__main__":
    backtest_df, current_df, bom_rank, mid_rank, bom_top4, mid_top4, df_cross = main()
    print(f"\nbacktest_df : {backtest_df.shape}")
    print(f"current_df  : {current_df.shape}")
    print(f"df_cross    : {df_cross.shape}")
