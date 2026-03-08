"""
data_engine.py — ETL Module (The "Eyes")
=========================================
Author : Srividya Manikandan (Data & Pipeline Lead)
Project: FX Decision Recommendation System for Indian Businesses

Provides a single entry-point function `get_final_data()` that returns
the complete historical + live exchange-rate DataFrame along with
ADF stationarity test results.
"""

import os
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolve path relative to *this* file so it works from any working directory
# data_engine.py lives in backend/, but data/ is at the project root (one level up)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BASE_DIR)
CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "cleaned_fx_data.csv")

# yfinance ticker symbols for the four currency pairs
TICKERS = {
    "USD": "USDINR=X",
    "GBP": "GBPINR=X",
    "EUR": "EURINR=X",
    "JPY": "JPYINR=X",
}

# Currencies in the order they appear in the CSV
CURRENCIES = ["USD", "GBP", "EUR", "JPY"]

# Rolling window for volatility calculation (matches preprocessing notebook)
VOLATILITY_WINDOW = 30


# ---------------------------------------------------------------------------
# Step 1 — Load Historical Data
# ---------------------------------------------------------------------------

def load_historical_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load the cleaned historical FX data from CSV.

    Returns a DataFrame indexed by Date with columns:
        EUR, GBP, JPY, USD, *_Return, *_Volatility
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.name = "Date"

    # Ensure the index is a proper DatetimeIndex (no timezone)
    df.index = pd.DatetimeIndex(df.index)

    print(f"[DATA ENGINE] Loaded {len(df)} rows of historical data "
          f"({df.index.min().date()} → {df.index.max().date()})")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Fetch Live Rates
# ---------------------------------------------------------------------------

def fetch_live_rates() -> dict:
    """
    Use yfinance to download the latest available price for each currency pair.

    Returns a dict like:
        {"USD": 85.12, "GBP": 107.45, "EUR": 92.30, "JPY": 55.60}

    The JPY value is already quoted per 100 JPY (matching the CSV convention).
    """
    live_rates = {}

    for currency, ticker in TICKERS.items():
        try:
            data = yf.download(ticker, period="1d", progress=False)

            if data.empty:
                warnings.warn(f"[DATA ENGINE] No data returned for {ticker}. "
                              "Market may be closed.")
                continue

            # yf.download returns a DataFrame; grab the last closing price
            close_col = data["Close"]
            # Handle multi-level columns (newer yfinance versions)
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            price = float(close_col.iloc[-1])

            # yfinance returns rate per 1 JPY (~0.58 INR).
            # Our CSV stores per 100 JPY (~57 INR), so multiply by 100.
            if currency == "JPY":
                price = price * 100.0

            live_rates[currency] = round(price, 4)
            print(f"[DATA ENGINE] Live {currency}/INR = {live_rates[currency]}")

        except Exception as exc:
            warnings.warn(f"[DATA ENGINE] Failed to fetch {ticker}: {exc}")

    return live_rates


# ---------------------------------------------------------------------------
# Step 3 — Append Live Row
# ---------------------------------------------------------------------------

def append_live_row(df: pd.DataFrame, live_rates: dict) -> pd.DataFrame:
    """
    Append the live rates as a new row dated today.

    If today's date already exists in the DataFrame, update it instead.
    Only the raw rate columns (USD, GBP, EUR, JPY) are filled here;
    derived columns are computed in the next step.
    """
    today = pd.Timestamp(date.today())

    if not live_rates:
        print("[DATA ENGINE] No live rates available — skipping append.")
        return df

    if today in df.index:
        # Update existing row with fresh live rates
        for cur, rate in live_rates.items():
            df.loc[today, cur] = rate
        print(f"[DATA ENGINE] Updated existing row for {today.date()}")
    else:
        # Create a new row with NaN for derived columns
        new_row = pd.DataFrame(
            {col: [np.nan] for col in df.columns},
            index=pd.DatetimeIndex([today], name="Date"),
        )
        for cur, rate in live_rates.items():
            new_row[cur] = rate

        df = pd.concat([df, new_row])
        df = df.sort_index()
        print(f"[DATA ENGINE] Appended new row for {today.date()}")

    return df


# ---------------------------------------------------------------------------
# Step 4 — Compute Derived Columns (Returns & Volatility)
# ---------------------------------------------------------------------------

def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate daily returns and 30-day rolling volatility for every
    currency so that the live row (and any gaps) are properly filled.
    """
    for cur in CURRENCIES:
        # Daily return = (price_t / price_{t-1}) – 1
        df[f"{cur}_Return"] = df[cur].pct_change()

        # 30-day rolling standard deviation of returns
        df[f"{cur}_Volatility"] = (
            df[f"{cur}_Return"]
            .rolling(window=VOLATILITY_WINDOW)
            .std()
        )

    print("[DATA ENGINE] Recomputed Returns & Volatility for all currencies.")
    return df


# ---------------------------------------------------------------------------
# Step 5 — ADF Stationarity Tests
# ---------------------------------------------------------------------------

def run_adf_tests(df: pd.DataFrame) -> dict:
    """
    Run the Augmented Dickey-Fuller test on each currency's Return series.

    Returns a dict of the form:
        {
            "USD": {"adf_stat": -59.12, "p_value": 0.0, "is_stationary": True},
            ...
        }
    """
    adf_results = {}

    for cur in CURRENCIES:
        col = f"{cur}_Return"
        series = df[col].dropna()

        if series.empty:
            warnings.warn(f"[DATA ENGINE] {col} is empty — skipping ADF test.")
            continue

        stat, pvalue, *_ = adfuller(series)
        is_stationary = pvalue < 0.05

        adf_results[cur] = {
            "adf_stat": round(stat, 4),
            "p_value": round(pvalue, 6),
            "is_stationary": is_stationary,
        }

        status = "STATIONARY ✓ (Ready for AI Modeling)" if is_stationary \
                 else "NON-STATIONARY ✗"
        print(f"[DATA ENGINE] ADF Test — {cur} Returns: "
              f"Stat={stat:.4f}, p={pvalue:.6f} → {status}")

    return adf_results


# ---------------------------------------------------------------------------
# Public API — get_final_data()
# ---------------------------------------------------------------------------

def get_final_data(csv_path: str = CSV_PATH):
    """
    Master function called by all downstream modules.

    Pipeline:
        1. Load historical cleaned_fx_data.csv
        2. Fetch live rates via yfinance
        3. Append / update the live row
        4. Recompute Returns & Volatility
        5. Run ADF stationarity tests

    Returns
    -------
    df : pd.DataFrame
        Complete FX dataset (historical + live), indexed by Date.
    adf_results : dict
        ADF test outcomes per currency.
    """
    print("=" * 60)
    print("  FX DATA ENGINE — Starting Pipeline")
    print("=" * 60)

    # 1. Historical data
    df = load_historical_data(csv_path)

    # 2. Live rates
    live_rates = fetch_live_rates()

    # 3. Append / update
    df = append_live_row(df, live_rates)

    # 4. Derived analytics
    df = compute_derived_columns(df)

    # 5. Stationarity check
    adf_results = run_adf_tests(df)

    print("=" * 60)
    print("  FX DATA ENGINE — Pipeline Complete")
    print(f"  Total rows: {len(df)} | Date range: "
          f"{df.index.min().date()} → {df.index.max().date()}")
    print("=" * 60)

    return df, adf_results


# ---------------------------------------------------------------------------
# Allow running the module directly for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, adf = get_final_data()
    print("\n--- Last 5 rows ---")
    print(df.tail())
    print("\n--- ADF Results ---")
    for cur, result in adf.items():
        print(f"  {cur}: {result}")
