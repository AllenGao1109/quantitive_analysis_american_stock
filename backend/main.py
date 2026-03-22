# backend/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

app = FastAPI(title="Quant Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/spx/monthly")
def get_spx_monthly(years: int = Query(default=5, ge=1, le=20)):
    df = yf.download("^GSPC", period=f"{years}y", interval="1mo", auto_adjust=True)
    df["monthly_return"] = df["Close"].pct_change() * 100
    df = df[["Close", "monthly_return"]].dropna().reset_index()
    df.columns = ["date", "close", "monthly_return"]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")

@app.get("/api/pension/rebalance")
def get_rebalance(
    stock_target: float = Query(default=0.6, ge=0.1, le=0.9),
    years: int = Query(default=5, ge=1, le=20),
    frequency: str = Query(default="monthly")
):
    df = yf.download("^GSPC", period=f"{years}y", interval="1mo", auto_adjust=True)
    df["monthly_return"] = df["Close"].pct_change() * 100
    df = df.dropna()

    stock_val = 1_000_000 * stock_target
    bond_val  = 1_000_000 * (1 - stock_target)
    records   = []

    for i, (date, row) in enumerate(df.iterrows()):
        stock_val *= (1 + row["monthly_return"] / 100)
        total = stock_val + bond_val
        drift = (stock_val / total) - stock_target

        should_rebalance = (
            frequency == "monthly" or
            (frequency == "quarterly" and i % 3 == 0)
        )
        rebalance_amt = 0.0
        if should_rebalance:
            rebalance_amt = drift * total
            stock_val = total * stock_target
            bond_val  = total * (1 - stock_target)

        records.append({
            "date":             date.strftime("%Y-%m-%d"),
            "portfolio_value":  round(total, 2),
            "stock_drift_pct":  round(drift * 100, 4),
            "rebalance_amount": round(-rebalance_amt, 2),
        })

    return {"target_stock": stock_target, "frequency": frequency, "data": records}
