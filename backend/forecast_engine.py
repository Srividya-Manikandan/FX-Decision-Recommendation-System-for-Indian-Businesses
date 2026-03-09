"""
forecast_engine.py — ML Forecasting Module (The "Future")
==========================================================
Author : Adwaitha (Forecasting Lead)
Project: FX Decision Recommendation System for Indian Businesses

Uses Facebook Prophet to generate 7-day exchange-rate forecasts with
95 % confidence intervals.  Every downstream module can call
`run_forecast()` to get a structured dictionary containing:
    • predicted rate (yhat)
    • upper bound   (yhat_upper) — 95th-percentile scenario
    • lower bound   (yhat_lower) —  5th-percentile scenario
    • trend direction (UP / DOWN)
    • full daily forecast table for visualization

Depends on
----------
data_engine.py  →  get_final_data()   (Srividya)
"""

import logging
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from prophet import Prophet

# Import Srividya's data pipeline
from data_engine import get_final_data

# Suppress noisy Prophet / cmdstanpy logs
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_CURRENCIES = ["USD", "GBP", "EUR", "JPY"]
DEFAULT_FORECAST_DAYS = 7        # 7-day outlook
MIN_TRAINING_ROWS = 60           # minimum datapoints for meaningful forecast
TRAINING_WINDOW = 365            # use last 1 year for faster training


# ---------------------------------------------------------------------------
# Helper — prepare Prophet-compatible DataFrame
# ---------------------------------------------------------------------------

def _prepare_prophet_df(df: pd.DataFrame, currency: str) -> pd.DataFrame:
    """
    Rename columns to Prophet's required format:
        ds  →  date
        y   →  exchange rate (e.g. USD/INR)

    Also drops NaN rows so Prophet doesn't choke.
    """
    if currency not in df.columns:
        raise ValueError(f"Currency '{currency}' not found in data. "
                         f"Available: {list(df.columns)}")

    pdf = df[[currency]].dropna().reset_index()
    pdf.columns = ["ds", "y"]

    # Prophet needs 'ds' as datetime without timezone
    pdf["ds"] = pd.to_datetime(pdf["ds"]).dt.tz_localize(None)

    return pdf


# ---------------------------------------------------------------------------
# Core — run_forecast()
# ---------------------------------------------------------------------------

def run_forecast(
    currency: str = "USD",
    days: int = DEFAULT_FORECAST_DAYS,
    df: pd.DataFrame = None,
) -> dict:
    """
    Generate a Prophet forecast for the given currency pair.

    Parameters
    ----------
    currency : str
        One of "USD", "GBP", "EUR", "JPY".
    days : int
        How many days into the future to forecast (default 7).
    df : pd.DataFrame, optional
        Pre-loaded data from `data_engine.get_final_data()`.
        If None, the function fetches fresh data automatically.

    Returns
    -------
    dict with keys:
        currency          – e.g. "USD"
        current_rate      – latest known rate
        predicted_rate    – yhat for the last forecast day
        forecast_upper    – yhat_upper (95 % CI upper bound)
        forecast_lower    – yhat_lower (95 % CI lower bound)
        trend             – "UP" or "DOWN"
        change_percent    – expected % change from current to predicted
        forecast_table    – list of dicts with daily forecast rows
                            [{ds, yhat, yhat_lower, yhat_upper}, …]
        training_rows     – how many rows were used for training
        status            – "success" or "error"
        message           – human-readable description
    """

    # ------------------------------------------------------------------
    # 1.  Load data (from Srividya) if not passed in
    # ------------------------------------------------------------------
    if df is None:
        df, _adf = get_final_data()

    # Validate currency
    currency = currency.upper()
    if currency not in SUPPORTED_CURRENCIES:
        return {
            "status": "error",
            "message": f"Unsupported currency '{currency}'. "
                       f"Choose from {SUPPORTED_CURRENCIES}",
        }

    # ------------------------------------------------------------------
    # 2.  Prepare training data
    # ------------------------------------------------------------------
    try:
        pdf = _prepare_prophet_df(df, currency)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if len(pdf) < MIN_TRAINING_ROWS:
        return {
            "status": "error",
            "message": f"Only {len(pdf)} rows available for {currency}; "
                       f"need at least {MIN_TRAINING_ROWS} for a reliable forecast.",
        }

    # Use only the most recent TRAINING_WINDOW days for speed
    training_data = pdf.tail(min(TRAINING_WINDOW, len(pdf))).copy()

    # ------------------------------------------------------------------
    # 3.  Train Prophet
    # ------------------------------------------------------------------
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95,          # 95 % confidence interval
        changepoint_prior_scale=0.05, # regularise changepoints
    )
    model.fit(training_data)

    print(f"[FORECAST ENGINE] Trained Prophet on {len(training_data)} rows "
          f"for {currency}/INR")

    # ------------------------------------------------------------------
    # 4.  Predict
    # ------------------------------------------------------------------
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # ------------------------------------------------------------------
    # 5.  Extract the forecast-only rows (future dates)
    # ------------------------------------------------------------------
    last_hist_date = training_data["ds"].max()
    future_only = forecast[forecast["ds"] > last_hist_date].copy()

    # If no future rows (edge case), take the last `days` rows
    if future_only.empty:
        future_only = forecast.tail(days).copy()

    # Final-day prediction
    last_row = future_only.iloc[-1]
    predicted_rate = round(float(last_row["yhat"]), 4)
    forecast_upper = round(float(last_row["yhat_upper"]), 4)
    forecast_lower = round(float(last_row["yhat_lower"]), 4)

    current_rate = round(float(training_data["y"].iloc[-1]), 4)

    trend = "UP" if predicted_rate > current_rate else "DOWN"
    change_pct = round(
        ((predicted_rate - current_rate) / current_rate) * 100, 4
    )

    # Build a clean daily table for the frontend / Streamlit charts
    forecast_table = []
    for _, row in future_only.iterrows():
        forecast_table.append({
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "yhat": round(float(row["yhat"]), 4),
            "yhat_lower": round(float(row["yhat_lower"]), 4),
            "yhat_upper": round(float(row["yhat_upper"]), 4),
        })

    # Build the full historical + forecast for plotting
    full_forecast = []
    for _, row in forecast.iterrows():
        full_forecast.append({
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "yhat": round(float(row["yhat"]), 4),
            "yhat_lower": round(float(row["yhat_lower"]), 4),
            "yhat_upper": round(float(row["yhat_upper"]), 4),
        })

    print(f"[FORECAST ENGINE] {currency}/INR  |  Current: {current_rate}  →  "
          f"Predicted: {predicted_rate}  ({'+' if change_pct >= 0 else ''}"
          f"{change_pct}%)  |  Trend: {trend}")
    print(f"[FORECAST ENGINE] 95% CI  →  Lower: {forecast_lower}  |  "
          f"Upper: {forecast_upper}")

    return {
        "status": "success",
        "currency": currency,
        "current_rate": current_rate,
        "predicted_rate": predicted_rate,
        "forecast_upper": forecast_upper,
        "forecast_lower": forecast_lower,
        "trend": trend,
        "change_percent": change_pct,
        "forecast_days": days,
        "forecast_table": forecast_table,
        "full_forecast": full_forecast,
        "training_rows": len(training_data),
        "message": (
            f"The {currency}/INR rate is currently {current_rate}. "
            f"Over the next {days} days it is forecast to move to "
            f"{predicted_rate} (trend: {trend}). "
            f"The 95% confidence band is "
            f"{forecast_lower} – {forecast_upper}."
        ),
    }


# ---------------------------------------------------------------------------
# Convenience — run_all_forecasts()
# ---------------------------------------------------------------------------

def run_all_forecasts(
    days: int = DEFAULT_FORECAST_DAYS,
) -> dict:
    """
    Run the forecast for ALL supported currency pairs in one go.

    Returns
    -------
    dict keyed by currency code, e.g.:
        {
            "USD": { ... forecast dict ... },
            "GBP": { ... },
            "EUR": { ... },
            "JPY": { ... },
        }
    """
    # Fetch data once, reuse across all pairs
    df, adf_results = get_final_data()

    results = {}
    for cur in SUPPORTED_CURRENCIES:
        print(f"\n{'─' * 50}")
        print(f"  Forecasting {cur}/INR …")
        print(f"{'─' * 50}")
        results[cur] = run_forecast(currency=cur, days=days, df=df)

    return results


# ---------------------------------------------------------------------------
# Quick-Test / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  FORECAST ENGINE — Standalone Test")
    print("=" * 60)

    # Forecast only USD for a quick smoke test
    result = run_forecast(currency="USD", days=7)

    if result["status"] == "success":
        print("\n✅  Forecast completed successfully!")
        print(f"   Currency        : {result['currency']}/INR")
        print(f"   Current Rate    : {result['current_rate']}")
        print(f"   Predicted Rate  : {result['predicted_rate']}")
        print(f"   Trend           : {result['trend']}  "
              f"({result['change_percent']:+.4f}%)")
        print(f"   95% CI Lower    : {result['forecast_lower']}")
        print(f"   95% CI Upper    : {result['forecast_upper']}")
        print(f"   Training Rows   : {result['training_rows']}")
        print(f"\n   7-Day Outlook:")
        for day in result["forecast_table"]:
            print(f"     {day['ds']}  →  ₹{day['yhat']:.2f}  "
                  f"( {day['yhat_lower']:.2f} – {day['yhat_upper']:.2f} )")
    else:
        print(f"\n❌  Forecast failed: {result['message']}")
