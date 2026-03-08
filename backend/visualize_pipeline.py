"""
visualize_pipeline.py — Data Pipeline Visualization
=====================================================
Author : Srividya Manikandan (Data & Pipeline Lead)
Project: FX Decision Recommendation System for Indian Businesses

Generates professional visualizations of the FX data pipeline output.
Saves all plots to outputs/plots/ for team presentations.

Usage:
    python backend/visualize_pipeline.py
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Add backend to path so we can import data_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_engine import get_final_data, CURRENCIES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Professional color palette
COLORS = {
    "USD": "#2563EB",   # Bold blue
    "GBP": "#7C3AED",   # Purple
    "EUR": "#059669",   # Emerald
    "JPY": "#DC2626",   # Red
}

CURRENCY_LABELS = {
    "USD": "USD/INR",
    "GBP": "GBP/INR",
    "EUR": "EUR/INR",
    "JPY": "JPY/INR (per 100¥)",
}

# Common plot styling
plt.rcParams.update({
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#E2E8F0",
    "text.color": "#E2E8F0",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Plot 1 — Historical Exchange Rate Trends (4-panel)
# ---------------------------------------------------------------------------

def plot_exchange_rate_trends(df: pd.DataFrame):
    """4-panel chart showing each currency's exchange rate over time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("Historical Exchange Rate Trends (2016–2026)",
                 fontsize=18, fontweight="bold", color="#F8FAFC", y=0.98)

    for ax, cur in zip(axes.flat, CURRENCIES):
        ax.plot(df.index, df[cur], color=COLORS[cur], linewidth=1.2, alpha=0.9)
        ax.fill_between(df.index, df[cur], alpha=0.1, color=COLORS[cur])
        ax.set_title(CURRENCY_LABELS[cur], fontsize=13, fontweight="bold",
                     color=COLORS[cur])
        ax.set_ylabel("Rate (INR)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Annotate latest value
        last_val = df[cur].dropna().iloc[-1]
        last_date = df[cur].dropna().index[-1]
        ax.annotate(f"₹{last_val:.2f}",
                    xy=(last_date, last_val),
                    fontsize=9, fontweight="bold",
                    color=COLORS[cur],
                    bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B",
                              ec=COLORS[cur], alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "exchange_rate_trends.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2 — Rolling Volatility (30-day)
# ---------------------------------------------------------------------------

def plot_rolling_volatility(df: pd.DataFrame):
    """Overlay chart showing 30-day rolling volatility for all currencies."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle("30-Day Rolling Volatility — Market \"Nervousness\" Indicator",
                 fontsize=16, fontweight="bold", color="#F8FAFC")

    for cur in CURRENCIES:
        vol_col = f"{cur}_Volatility"
        ax.plot(df.index, df[vol_col], color=COLORS[cur],
                linewidth=1.0, alpha=0.85, label=CURRENCY_LABELS[cur])

    # Highlight high-volatility threshold
    ax.axhline(y=0.008, color="#FBBF24", linestyle="--", linewidth=1,
               alpha=0.7, label="High Volatility Threshold")

    ax.set_ylabel("Volatility (Std Dev of Returns)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, facecolor="#1E293B",
              edgecolor="#475569")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "rolling_volatility.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3 — Returns Distribution (Histograms)
# ---------------------------------------------------------------------------

def plot_returns_distribution(df: pd.DataFrame):
    """Histogram + KDE for each currency's daily returns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Daily Returns Distribution — Stationarity Validation",
                 fontsize=16, fontweight="bold", color="#F8FAFC", y=0.98)

    for ax, cur in zip(axes.flat, CURRENCIES):
        returns = df[f"{cur}_Return"].dropna()
        ax.hist(returns, bins=80, color=COLORS[cur], alpha=0.7,
                edgecolor="none", density=True)

        # Overlay normal distribution curve
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 200)
        ax.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) *
                np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                color="#F8FAFC", linewidth=1.5, alpha=0.8,
                label=f"Normal (μ={mu:.5f})")

        ax.set_title(f"{CURRENCY_LABELS[cur]} Returns", fontsize=12,
                     fontweight="bold", color=COLORS[cur])
        ax.set_xlabel("Daily Return", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, facecolor="#1E293B", edgecolor="#475569")
        ax.grid(True, linestyle="--", alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "returns_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4 — ADF Stationarity Summary
# ---------------------------------------------------------------------------

def plot_adf_summary(adf_results: dict):
    """Horizontal bar chart summarizing ADF test statistics."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("ADF Stationarity Test Results",
                 fontsize=16, fontweight="bold", color="#F8FAFC")

    currencies = list(adf_results.keys())
    stats = [adf_results[c]["adf_stat"] for c in currencies]
    colors_list = [COLORS[c] for c in currencies]

    bars = ax.barh(currencies, stats, color=colors_list, height=0.5,
                   edgecolor="none", alpha=0.85)

    # Critical value line (approx. -2.86 for 5% significance)
    ax.axvline(x=-2.86, color="#FBBF24", linestyle="--", linewidth=1.5,
               label="5% Critical Value (-2.86)")

    # Labels on bars
    for bar, stat, cur in zip(bars, stats, currencies):
        p = adf_results[cur]["p_value"]
        label = f"  Stat: {stat:.2f} | p: {p:.6f} ✓"
        ax.text(stat + 0.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10, color="#E2E8F0",
                fontweight="bold")

    ax.set_xlabel("ADF Statistic (more negative = more stationary)",
                  fontsize=11)
    ax.legend(loc="upper right", fontsize=9, facecolor="#1E293B",
              edgecolor="#475569")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "adf_stationarity_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  FX DATA PIPELINE — Generating Visualizations")
    print("=" * 60)

    # Run the data pipeline
    df, adf_results = get_final_data()

    # Generate all plots
    plot_exchange_rate_trends(df)
    plot_rolling_volatility(df)
    plot_returns_distribution(df)
    plot_adf_summary(adf_results)

    print("\n" + "=" * 60)
    print(f"  All plots saved to: {OUTPUT_DIR}")
    print("=" * 60)
