# 📈 SPX Month Strategy

> A research-driven, regime-aware monthly trading framework for the S&P 500, combining statistical learning, risk modeling, and walk-forward validation.

---

## 🚀 Overview

This project implements a **fully reproducible quantitative research + deployment pipeline** for monthly SPX timing.

It integrates:

- 📊 Macro + market data (FRED, Stooq)
- 🧠 Machine learning (Logistic / Elastic Net)
- 🌪 Regime & volatility modeling (HMM, OU, GARCH-style features)
- 🔄 Walk-forward validation (strict OOS)
- ⚖️ Risk-aware position sizing
- 📉 Backtesting + dashboard visualization

👉 The goal is **robust, regime-adaptive return generation**, not overfitting.

---

## 🧠 Modeling Philosophy

This project is heavily inspired by a combination of academic and practitioner literature:

### 1. Return Predictability & Macro Signals
- **Campbell & Thompson (2008)** – Predicting excess stock returns
- **Cochrane (2011)** – Discount rate variation
- **Fama & French** – Risk premia decomposition

👉 Motivation:
Markets are *not purely random* at monthly horizons — macro and volatility states matter.

---

### 2. Regime Switching & State Models
- **Hamilton (1989)** – Regime switching models
- **Rabiner (1989)** – Hidden Markov Models
- **Ang & Timmermann (2012)** – Regime changes in asset allocation

👉 Implementation:
- HMM-based stress probability
- Regime-aware features (risk-on / risk-off)

---

### 3. Volatility & Tail Risk Modeling
- **Engle (1982)** – ARCH / GARCH
- **Bollerslev (1986)** – GARCH extensions
- **Barndorff-Nielsen & Shephard (2004)** – realized volatility

👉 Implementation:
- GARCH-style volatility normalization
- Tail-risk proxies
- Hawkes-like jump intensity (VIX-based)

---

### 4. Mean Reversion & OU Processes
- **Ornstein-Uhlenbeck (OU)** processes
- Widely used in statistical arbitrage

👉 Implementation:
- OU-style z-score features
- Reversion pressure signals

---

### 5. Ensemble & Regularized Learning
- Logistic regression (interpretable baseline)
- Elastic Net (sparse + robust)

👉 Inspired by:
- **Hastie, Tibshirani, Friedman (ESL)**
- Modern ML practice in finance

---

### 6. Walk-Forward Validation (CRITICAL)
Unlike many academic papers:

❌ No random split  
❌ No leakage  
✅ Strict expanding window  
✅ True out-of-sample testing  

---

## 🏗️ Architecture

```
Data → Feature Engineering → v9.2 Features → Walk-forward Models
     → Position Construction → Backtest → Dashboard
```

---

## ⚙️ Strategy Design

### Two Trading Segments

| Segment | Description |
|--------|------------|
| **BOM** | Beginning-of-month signal |
| **MID** | Mid-month continuation/reversal |

---

### Position Families

#### 1. Base
- Direct mapping from logit → position

#### 2. SoftRisk
- Position × risk multiplier
- Penalizes:
  - volatility spikes
  - regime stress
  - drawdowns

#### 3. LogitStrength
- Uses percentile ranking of model confidence

---

### Position Types

- **LO** → Long only  
- **LS** → Long / Short  

---

## 📊 Backtest Framework

- Monthly frequency
- No leverage
- Fully out-of-sample
- Metrics:
  - Sharpe
  - Max Drawdown
  - NAV curve
  - Monthly heatmap

---

## 🖥 Dashboard

Built with **Streamlit + Plotly**

### Features:
- Strategy comparison table
- NAV curves
- Position history
- Monthly return heatmap
- Current signal recommendation

---

## 🧪 How to Run

### 1. Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Dashboard

```bash
streamlit run dashboard_app_month.py
```

---

## 📦 Project Structure

```
.
├── dashboard_app_month.py     # UI
├── spx_month_strategy.py      # Core engine
├── requirements.txt
└── README.md
```

---

## ⚠️ Practical Notes

### Data Reliability
- Uses public APIs → may fail
- Recommend caching locally

### Overfitting Control
- Walk-forward only
- No peeking

### Deployment Tip
- Add CSV fallback for production

---

## 🔮 Future Work

- Transaction costs
- Live trading API
- Model persistence
- Multi-asset extension
- Reinforcement learning (carefully applied)

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.

Not financial advice.

---

## ✍️ Author Note

This system reflects a hybrid approach:

> "Combine simple models + strong structure + strict validation"

instead of:

> "Overcomplicated models + weak validation"

The edge comes from:
- regime awareness
- risk control
- disciplined backtesting

—not model complexity.

