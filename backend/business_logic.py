"""
business_logic.py — Strategy Module (The "Advice")
====================================================
Author : Aadhithya Bharathi (Exposure Lead)
Project: FX Decision Recommendation System for Indian Businesses

Translates technical FX analytics (forecast, risk, volatility) into
actionable business decisions for Importers, Exporters, and IT Firms.

Functions
---------
- get_business_exposure   : Profile a deal's FX exposure
- calculate_profit_at_risk: Show best/worst INR outcome using forecast bounds
- get_recommendation      : Prescriptive Hedge / Wait / Split decision
- generate_sensitivity_matrix : What-if matrix across rate changes
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Business Exposure Profiling
# ---------------------------------------------------------------------------

def get_business_exposure(deal_size: float,
                          business_type: str = "Importer",
                          current_rate: float = 85.0) -> dict:
    """
    Compute the FX exposure profile for a given deal.

    Parameters
    ----------
    deal_size      : Transaction amount in foreign currency (e.g. USD)
    business_type  : "Importer", "Exporter", or "IT Firm"
    current_rate   : Current INR exchange rate per 1 unit of foreign currency

    Returns
    -------
    dict with base_inr_value, sensitivity_per_rupee, risk_zone, and message
    """
    base_inr_value = deal_size * current_rate
    sensitivity_per_rupee = deal_size      # ₹1 rate move = deal_size INR impact

    # Risk zone thresholds (from Sprint-1 notebook)
    if deal_size >= 500_000:
        zone, priority = "Danger", "High"
        message = "Critical Exposure. Active hedging is mandatory."
    elif deal_size >= 100_000:
        zone, priority = "Warning", "Medium"
        message = "Moderate Exposure. Consider partial hedging."
    else:
        zone, priority = "Safe", "Low"
        message = "Low impact. Spot conversion is acceptable."

    # Business-type specific note
    if business_type == "Importer":
        direction_note = "A rising rate INCREASES your INR cost."
    elif business_type == "Exporter":
        direction_note = "A falling rate DECREASES your INR revenue."
    else:  # IT Firm
        direction_note = "A falling rate REDUCES your INR receivables."

    return {
        "deal_size": deal_size,
        "business_type": business_type,
        "current_rate": round(current_rate, 4),
        "base_inr_value": round(base_inr_value, 2),
        "sensitivity_per_rupee": round(sensitivity_per_rupee, 2),
        "risk_zone": zone,
        "risk_priority": priority,
        "direction_note": direction_note,
        "message": message,
    }


# ---------------------------------------------------------------------------
# 2. Profit-at-Risk (PaR) Calculation
# ---------------------------------------------------------------------------

def calculate_profit_at_risk(deal_size: float,
                             business_type: str,
                             current_rate: float,
                             forecast_upper: float,
                             forecast_lower: float) -> dict:
    """
    Quantify best-case and worst-case financial outcomes using
    Prophet's 95% confidence interval (yhat_upper & yhat_lower).

    Parameters
    ----------
    deal_size       : Transaction amount in foreign currency
    business_type   : "Importer" or "Exporter" (or "IT Firm" treated as Exporter)
    current_rate    : Today's spot rate
    forecast_upper  : Prophet yhat_upper (high-end rate forecast)
    forecast_lower  : Prophet yhat_lower (low-end rate forecast)

    Returns
    -------
    dict with best/worst case rates, INR amounts, and PaR summary
    """
    base_cost = deal_size * current_rate
    upper_cost = deal_size * forecast_upper
    lower_cost = deal_size * forecast_lower

    # Importer perspective: higher rate = higher cost = LOSS
    # Exporter / IT Firm perspective: higher rate = higher revenue = GAIN
    is_importer = business_type == "Importer"

    if is_importer:
        best_case_rate   = forecast_lower   # lower rate = cheaper import
        worst_case_rate  = forecast_upper   # higher rate = costlier import
        best_case_inr    = lower_cost
        worst_case_inr   = upper_cost
        profit_at_risk   = upper_cost - base_cost   # extra cost in worst case
    else:
        best_case_rate   = forecast_upper   # higher rate = more INR revenue
        worst_case_rate  = forecast_lower   # lower rate = less INR revenue
        best_case_inr    = upper_cost
        worst_case_inr   = lower_cost
        profit_at_risk   = base_cost - lower_cost   # lost revenue in worst case

    pct_change_worst = ((worst_case_rate - current_rate) / current_rate) * 100

    # Human-readable warning
    par_lakhs = abs(profit_at_risk) / 100_000
    if is_importer:
        warning = (f"If the rate rises to ₹{worst_case_rate:.2f}, your import "
                   f"cost increases by ₹{abs(profit_at_risk):,.0f} "
                   f"(~₹{par_lakhs:.2f} Lakhs).")
    else:
        warning = (f"If the rate drops to ₹{worst_case_rate:.2f}, your revenue "
                   f"decreases by ₹{abs(profit_at_risk):,.0f} "
                   f"(~₹{par_lakhs:.2f} Lakhs).")

    return {
        "current_rate": round(current_rate, 4),
        "best_case_rate": round(best_case_rate, 4),
        "worst_case_rate": round(worst_case_rate, 4),
        "base_inr_value": round(base_cost, 2),
        "best_case_inr": round(best_case_inr, 2),
        "worst_case_inr": round(worst_case_inr, 2),
        "profit_at_risk_inr": round(abs(profit_at_risk), 2),
        "profit_at_risk_lakhs": round(par_lakhs, 2),
        "worst_case_pct_change": round(pct_change_worst, 2),
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# 3. Prescriptive Recommendation Engine
# ---------------------------------------------------------------------------

def get_recommendation(deal_size: float,
                       business_type: str,
                       risk_score: float,
                       risk_level: str,
                       forecast_trend: str,
                       current_rate: float,
                       predicted_rate: float) -> dict:
    """
    The ultimate decision engine — maps risk + forecast into a business action.

    Decision Matrix
    ---------------
    High Risk + INR Weakening (rate UP for Importer)  → HEDGE IMMEDIATELY
    High Risk + INR Strengthening                     → SPLIT HEDGE (50-50)
    Medium Risk + Unfavorable Trend                   → SPLIT HEDGE (50-50)
    Medium Risk + Favorable Trend                     → WATCHFUL WAITING
    Low Risk + Favorable Trend                        → DELAY CONVERSION
    Low Risk + Unfavorable Trend                      → WATCHFUL WAITING

    Parameters
    ----------
    deal_size       : Foreign currency amount
    business_type   : "Importer", "Exporter", or "IT Firm"
    risk_score      : Composite 0–100 score from risk_engine / fx_engine
    risk_level      : "Low", "Medium", or "High"
    forecast_trend  : "UP" or "DOWN" (INR rate direction)
    current_rate    : Spot rate today
    predicted_rate  : Prophet yhat (7-day forecast)

    Returns
    -------
    dict with action, urgency, reasoning, and financial context
    """
    is_importer = business_type == "Importer"
    rate_change = predicted_rate - current_rate
    rate_change_pct = (rate_change / current_rate) * 100
    inr_impact = deal_size * abs(rate_change)

    # Is the trend unfavorable for THIS business type?
    # Importer: rate going UP is bad (costs more)
    # Exporter / IT: rate going DOWN is bad (earns less)
    unfavorable = (is_importer and forecast_trend == "UP") or \
                  (not is_importer and forecast_trend == "DOWN")

    # ---- Decision Logic ----
    if risk_level == "High" and unfavorable:
        action = "HEDGE / CONVERT IMMEDIATELY"
        urgency = "CRITICAL"
        color = "red"
        reasoning = (
            f"Risk Score is {risk_score}/100 (High) and the forecast shows "
            f"the rate moving AGAINST your position by {abs(rate_change_pct):.2f}%. "
            f"Potential INR impact: ₹{inr_impact:,.0f}. "
            f"Secure forward contracts or convert your full exposure NOW."
        )
        hedge_pct = 100

    elif risk_level == "High" and not unfavorable:
        action = "SPLIT HEDGING (75-25)"
        urgency = "HIGH"
        color = "orange"
        reasoning = (
            f"Risk Score is {risk_score}/100 (High) but the trend is currently "
            f"favorable. Hedge 75% of your exposure now and hold 25% to benefit "
            f"from the positive trend."
        )
        hedge_pct = 75

    elif risk_level == "Medium" and unfavorable:
        action = "SPLIT HEDGING (50-50)"
        urgency = "MODERATE"
        color = "yellow"
        reasoning = (
            f"Risk Score is {risk_score}/100 (Medium) with an unfavorable trend "
            f"({forecast_trend}). Convert 50% now to lock in the current rate, "
            f"and watch the remaining 50% for a better window."
        )
        hedge_pct = 50

    elif risk_level == "Medium" and not unfavorable:
        action = "WATCHFUL WAITING"
        urgency = "LOW"
        color = "blue"
        reasoning = (
            f"Risk Score is {risk_score}/100 (Medium) but the trend favors your "
            f"position. No immediate action needed — monitor daily and act if "
            f"conditions change."
        )
        hedge_pct = 25

    elif risk_level == "Low" and unfavorable:
        action = "WATCHFUL WAITING"
        urgency = "LOW"
        color = "blue"
        reasoning = (
            f"Risk Score is {risk_score}/100 (Low). Although the trend is slightly "
            f"unfavorable, overall risk remains manageable. Monitor and convert "
            f"in the next 2–3 days if the trend persists."
        )
        hedge_pct = 25

    else:  # Low risk + favorable
        action = "DELAY CONVERSION"
        urgency = "NONE"
        color = "green"
        reasoning = (
            f"Risk Score is {risk_score}/100 (Low) and the market is moving in "
            f"your favor. Optimal conditions — no hedging needed. Use spot "
            f"conversion when your payment is due."
        )
        hedge_pct = 0

    # Calculate recommended hedge amount
    hedge_amount_fx = deal_size * (hedge_pct / 100)
    hedge_amount_inr = hedge_amount_fx * current_rate

    return {
        "action": action,
        "urgency": urgency,
        "color": color,
        "hedge_percentage": hedge_pct,
        "hedge_amount_fx": round(hedge_amount_fx, 2),
        "hedge_amount_inr": round(hedge_amount_inr, 2),
        "reasoning": reasoning,
        "rate_change_inr": round(rate_change, 4),
        "rate_change_pct": round(rate_change_pct, 2),
        "potential_impact_inr": round(inr_impact, 2),
    }


# ---------------------------------------------------------------------------
# 4. Sensitivity Matrix (What-If Analysis)
# ---------------------------------------------------------------------------

def generate_sensitivity_matrix(deal_size: float,
                                business_type: str,
                                current_rate: float) -> list:
    """
    Generate a what-if matrix showing profit/loss across rate changes.

    Shows the impact of ±1%, ±2%, ±3%, and ±5% rate movements so the
    business owner can visualise the range of financial outcomes.

    Returns
    -------
    list of dicts, each with: scenario, pct_change, new_rate, inr_value,
    gain_loss, and impact_label
    """
    is_importer = business_type == "Importer"
    base_cost = deal_size * current_rate

    pct_changes = [-5, -3, -2, -1, 0, +1, +2, +3, +5]
    matrix = []

    for pct in pct_changes:
        new_rate = current_rate * (1 + pct / 100)
        new_cost = deal_size * new_rate
        diff = new_cost - base_cost

        # For Importer: positive diff = extra cost (loss)
        # For Exporter: negative diff = less revenue (loss)
        if is_importer:
            gain_loss = -diff    # negative = loss for importer
            label = "LOSS" if gain_loss < 0 else ("GAIN" if gain_loss > 0 else "NO CHANGE")
        else:
            gain_loss = diff     # positive = gain for exporter
            label = "LOSS" if gain_loss < 0 else ("GAIN" if gain_loss > 0 else "NO CHANGE")

        # Scenario name
        if pct == 0:
            scenario = "Current (Base)"
        elif pct > 0:
            scenario = f"Rate rises {pct}%"
        else:
            scenario = f"Rate falls {abs(pct)}%"

        matrix.append({
            "scenario": scenario,
            "pct_change": pct,
            "new_rate": round(new_rate, 4),
            "inr_value": round(new_cost, 2),
            "gain_loss_inr": round(gain_loss, 2),
            "gain_loss_lakhs": round(gain_loss / 100_000, 2),
            "impact_label": label,
        })

    return matrix


# ---------------------------------------------------------------------------
# Standalone Test — Run with: python business_logic.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import sys
    # Fix Windows console encoding for ₹ symbol
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 65)
    print("  BUSINESS LOGIC MODULE — Standalone Test")
    print("  Author: Aadhithya Bharathi | Exposure Lead")
    print("=" * 65)

    # Sample inputs (simulate what fx_engine / data_engine would provide)
    DEAL_SIZE       = 100_000        # $100,000 deal
    BIZ_TYPE        = "Importer"
    CURRENT_RATE    = 85.50          # ₹85.50 per USD
    FORECAST_UPPER  = 86.80          # Prophet yhat_upper
    FORECAST_LOWER  = 84.20          # Prophet yhat_lower
    PREDICTED_RATE  = 86.10          # Prophet yhat (7-day)
    RISK_SCORE      = 62.5           # from risk engine
    RISK_LEVEL      = "Medium"
    TREND           = "UP"

    # ---- 1. Exposure Profile ----
    print("\n" + "-" * 65)
    print("  1. BUSINESS EXPOSURE PROFILE")
    print("-" * 65)
    exposure = get_business_exposure(DEAL_SIZE, BIZ_TYPE, CURRENT_RATE)
    for k, v in exposure.items():
        print(f"  {k:>25s} : {v}")

    # ---- 2. Profit-at-Risk ----
    print("\n" + "-" * 65)
    print("  2. PROFIT-AT-RISK (PaR)")
    print("-" * 65)
    par = calculate_profit_at_risk(
        DEAL_SIZE, BIZ_TYPE, CURRENT_RATE, FORECAST_UPPER, FORECAST_LOWER
    )
    for k, v in par.items():
        print(f"  {k:>25s} : {v}")

    # ---- 3. Recommendation ----
    print("\n" + "-" * 65)
    print("  3. PRESCRIPTIVE RECOMMENDATION")
    print("-" * 65)
    rec = get_recommendation(
        DEAL_SIZE, BIZ_TYPE, RISK_SCORE, RISK_LEVEL, TREND,
        CURRENT_RATE, PREDICTED_RATE
    )
    for k, v in rec.items():
        print(f"  {k:>25s} : {v}")

    # ---- 4. Sensitivity Matrix ----
    print("\n" + "-" * 65)
    print("  4. SENSITIVITY MATRIX (What-If Analysis)")
    print("-" * 65)
    matrix = generate_sensitivity_matrix(DEAL_SIZE, BIZ_TYPE, CURRENT_RATE)

    header = f"  {'Scenario':<22s} {'Rate':>9s} {'INR Value':>14s} {'Gain/Loss':>14s} {'Status':>10s}"
    print(header)
    print("  " + "-" * len(header))
    for row in matrix:
        print(f"  {row['scenario']:<22s} "
              f"₹{row['new_rate']:>7.2f} "
              f"₹{row['inr_value']:>12,.0f} "
              f"₹{row['gain_loss_inr']:>+12,.0f} "
              f"{row['impact_label']:>10s}")

    print("\n" + "=" * 65)
    print("  TEST COMPLETE — All functions executed successfully ✓")
    print("=" * 65)
