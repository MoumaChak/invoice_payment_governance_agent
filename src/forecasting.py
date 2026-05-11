from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CAPTURE_RATE, OUTPUTS_DIR


def _has(module: str) -> bool:
    if module == "prophet" and os.getenv("ENABLE_PROPHET", "0") != "1":
        return False
    return importlib.util.find_spec(module) is not None


def create_leakage_forecast(df: pd.DataFrame, outputs_dir: Path = OUTPUTS_DIR, periods: int = 6) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    data["sap_payment_release_date"] = pd.to_datetime(data["sap_payment_release_date"], errors="coerce")
    early = data[data["early_payment_flag"] == 1].copy()
    monthly = early.groupby(early["sap_payment_release_date"].dt.to_period("M")).agg(
        early_payment_count=("invoice_id", "count"),
        early_payment_value=("invoice_amount", "sum"),
        avg_days_early=("days_paid_early", "mean"),
    ).reset_index()
    monthly["month"] = monthly["sap_payment_release_date"].astype(str)
    monthly = monthly.drop(columns=["sap_payment_release_date"])
    monthly["ds"] = pd.to_datetime(monthly["month"] + "-01")
    if monthly.empty:
        out = pd.DataFrame(columns=["ds", "early_payment_count", "early_payment_value", "forecast_type", "retained_value_intervention"])
        out.to_csv(outputs_dir / "leakage_forecast.csv", index=False)
        return out

    hist = monthly[["ds", "early_payment_count", "early_payment_value"]].copy()
    hist["forecast_type"] = "Historical"
    hist["retained_value_intervention"] = hist["early_payment_value"] * CAPTURE_RATE

    future_dates = pd.date_range(hist["ds"].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS")

    if _has("prophet") and len(monthly) >= 6:
        try:
            from prophet import Prophet
            p_val = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            train_val = monthly[["ds", "early_payment_value"]].rename(columns={"early_payment_value": "y"})
            p_val.fit(train_val)
            fc_val = p_val.predict(pd.DataFrame({"ds": future_dates}))["yhat"].clip(lower=0).values
            p_cnt = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            train_cnt = monthly[["ds", "early_payment_count"]].rename(columns={"early_payment_count": "y"})
            p_cnt.fit(train_cnt)
            fc_cnt = p_cnt.predict(pd.DataFrame({"ds": future_dates}))["yhat"].clip(lower=0).values
        except Exception:
            fc_val, fc_cnt = None, None
    else:
        fc_val, fc_cnt = None, None

    if fc_val is None:
        last_val = monthly["early_payment_value"].tail(3).mean()
        last_cnt = monthly["early_payment_count"].tail(3).mean()
        growth = 1 + np.linspace(0.01, 0.06, periods)
        fc_val = last_val * growth
        fc_cnt = last_cnt * growth

    forecast = pd.DataFrame({
        "ds": future_dates,
        "early_payment_count": np.round(fc_cnt).astype(int),
        "early_payment_value": np.round(fc_val, 2),
        "forecast_type": "Forecast - No Intervention",
    })
    forecast["retained_value_intervention"] = forecast["early_payment_value"] * CAPTURE_RATE
    result = pd.concat([hist, forecast], ignore_index=True)
    result["working_capital_leakage"] = result["early_payment_value"]
    result["future_leakage_without_intervention"] = result["early_payment_value"]
    result["future_retained_with_intervention"] = result["retained_value_intervention"]
    result.to_csv(outputs_dir / "leakage_forecast.csv", index=False)
    return result
