"""
risk_engine.py — Quant Module (The "Safety")
=============================================
Author : Kanishkhan (Risk Lead)
Project: FX Decision Recommendation System for Indian Businesses

This module quantifies market danger using:
1. 30-day Rolling Volatility
2. Z-Score Anomaly Detection
3. Value-at-Risk (VaR)
"""

import pandas as pd
import numpy as np

def calculate_risk_metrics(df: pd.DataFrame, currency: str = "USD") -> dict:
    """
    Calculates technical risk metrics for a specific currency pair.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing historical and live exchange rates (from data_engine).
    currency : str
        The currency code (e.g., 'USD', 'EUR', 'GBP').
        
    Returns:
    --------
    dict
        A dictionary containing Volatility, Z-Score, Anomaly flag, and VaR.
    """
    # 1. Ensure we have the required series
    if currency not in df.columns:
        raise ValueError(f"Currency {currency} not found in the dataset.")
    
    series = df[currency].dropna()
    returns = df[f"{currency}_Return"].dropna()
    
    # --- Step 1: 30-day Rolling Volatility (Standard Deviation of Rates) ---
    # The prompt explicitly asks for volatility of "the rates".
    rolling_vol = series.rolling(window=30).std()
    current_vol = rolling_vol.iloc[-1]
    
    # --- Step 2: Z-Score Anomaly Detection ---
    # Formula: (Current Rate - 30-day Average) / 30-day StdDev
    rolling_mean = series.rolling(window=30).mean()
    current_rate = series.iloc[-1]
    last_mean = rolling_mean.iloc[-1]
    last_std = rolling_vol.iloc[-1]
    
    if last_std == 0 or np.isnan(last_std):
        z_score = 0
    else:
        z_score = (current_rate - last_mean) / last_std
    
    is_anomaly = abs(z_score) > 2.0
    
    # --- Step 3: Value-at-Risk (VaR) ---
    # Find the 5th percentile of historical daily drops (negative returns).
    # We convert this percentile (e.g., -0.01) into an absolute INR drop.
    var_return_threshold = np.percentile(returns, 5)
    inr_loss_amount = abs(var_return_threshold * current_rate)
    
    # The message as requested: "There is a 5% chance you could lose more than X amount of INR by tomorrow."
    var_message = f"There is a 5% chance you could lose more than {inr_loss_amount:.2f} INR by tomorrow."
    
    return {
        "currency": currency,
        "current_rate": round(current_rate, 4),
        "30d_volatility": round(current_vol, 4),
        "z_score": round(z_score, 2),
        "is_anomaly": is_anomaly,
        "var_95_percentile": round(var_return_threshold, 6),
        "inr_loss_amount": round(inr_loss_amount, 2),
        "var_message": var_message
    }

def get_risk_report(df: pd.DataFrame):
    """
    Generates a risk report for all major currencies.
    """
    currencies = ["USD", "GBP", "EUR", "JPY"]
    report = {}
    for cur in currencies:
        try:
            report[cur] = calculate_risk_metrics(df, cur)
        except Exception as e:
            print(f"[RISK ENGINE] Error calculating metrics for {cur}: {e}")
    return report

if __name__ == "__main__":
    # Quick Test
    from data_engine import get_final_data
    df, _ = get_final_data()
    usd_risk = calculate_risk_metrics(df, "USD")
    print("\n--- USD Risk Metrics ---")
    for k, v in usd_risk.items():
        print(f"{k}: {v}")
