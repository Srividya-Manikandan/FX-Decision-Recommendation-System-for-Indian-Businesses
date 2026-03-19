"""
api_bridge.py — Flask API Bridge
==================================
Author : Adarsh (Integration Lead)
Project: FX Decision Recommendation System for Indian Businesses

Bridges the Python analytical engines to the React frontend via REST API.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import numpy as np

# Ensure backend modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fx_engine import FXEngine
from exposure_engine import ExposureEngine
from risk_engine import calculate_risk_metrics
from forecast_engine import run_forecast, run_all_forecasts
from business_logic import (
    get_business_exposure,
    calculate_profit_at_risk,
    get_recommendation,
    generate_sensitivity_matrix
)


# Recursive sanitizer: convert all numpy types to native Python
def sanitize(obj):
    """Recursively convert numpy types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.ndarray,)):
        return sanitize(obj.tolist())
    # Handle Python float NaN too
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


app = Flask(__name__)
CORS(app)

# Initialize engines once
engine = FXEngine()
exposure_engine = ExposureEngine()

# Paths
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'plots')


# ---------------------------------------------------------------------------
# 1. Main Dashboard Endpoint
# ---------------------------------------------------------------------------
@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    """
    Returns consolidated FX dashboard data.
    Query params:
      - date: optional target date (YYYY-MM-DD)
      - include_analysis: 'true' to include historical/volatility/correlation data
      - horizon: forecast horizon in days (default 7)
    """
    target_date = request.args.get('date', None)
    include_analysis = request.args.get('include_analysis', 'true').lower() == 'true'
    horizon = int(request.args.get('horizon', 7))

    try:
        data = engine.get_full_dashboard(
            include_analysis=include_analysis,
            target_date=target_date,
            forecast_days=horizon
        )

        # Enrich with risk engine metrics for each currency
        if engine.df_master is not None:
            risk_details = {}
            for currency in ['USD', 'GBP', 'EUR', 'JPY']:
                try:
                    metrics = calculate_risk_metrics(
                        engine.df_master, currency, exposure_usd=100000
                    )
                    risk_details[currency] = {
                        'z_score': metrics['z_score'],
                        'is_anomaly': metrics['is_anomaly'],
                        'var_95_percentile': metrics['var_95_percentile'],
                        'inr_loss_amount': metrics['inr_loss_amount'],
                        'var_message': metrics['var_message'],
                        '30d_volatility': metrics['30d_volatility'],
                        'score': metrics['score'],
                        'level': metrics['level']
                    }
                except Exception as e:
                    risk_details[currency] = {'error': str(e)}

            data['risk_details'] = risk_details

        return jsonify(sanitize(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# 2. Exposure Calculator Endpoint
# ---------------------------------------------------------------------------
@app.route('/api/calculate-exposure', methods=['POST'])
def calculate_exposure():
    """
    Calculates scenario analysis and sensitivity for a given deal.
    Body: { amount, currency, type }
    """
    body = request.get_json()
    amount = float(body.get('amount', 100000))
    currency = body.get('currency', 'USD')
    business_type = body.get('type', 'Importer')

    try:
        engine.run_preprocessing()
        if engine.df_master is None:
            return jsonify({'error': 'Data engine not ready'}), 500

        current_rate = float(engine.df_master[currency].iloc[-1])

        scenarios = exposure_engine.calculate_scenarios(amount, current_rate, business_type)
        sensitivity = exposure_engine.get_sensitivity(amount, business_type)

        return jsonify(sanitize({
            'current_rate': round(current_rate, 4),
            'scenarios': scenarios,
            'sensitivity': sensitivity
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# 3. Business Recommendation Endpoint
# ---------------------------------------------------------------------------
@app.route('/api/business-recommendation', methods=['POST'])
def business_recommendation():
    """
    Full prescriptive recommendation with break-even, margin analysis, and hedge advice.
    Body: { amount, currency, type, targetMargin }
    """
    body = request.get_json()
    deal_size = float(body.get('amount', 100000))
    currency = body.get('currency', 'USD')
    business_type = body.get('type', 'Importer')
    target_margin = float(body.get('targetMargin', 5.0))
    horizon = int(body.get('horizon', 7))

    try:
        engine.run_preprocessing()
        if engine.df_master is None:
            return jsonify({'error': 'Data engine not ready'}), 500

        current_rate = float(engine.df_master[currency].iloc[-1])

        # Risk metrics
        risk_metrics = calculate_risk_metrics(engine.df_master, currency, exposure_usd=deal_size)

        # Forecast
        forecast = engine.get_forecast(currency=currency, days=horizon)
        if forecast is None or 'error' in forecast:
            return jsonify({'error': 'Forecast engine not ready'}), 500

        predicted_rate = forecast['predicted_rate']
        trend = forecast['trend']

        # Break-even calculation
        # For Importer: break-even is the rate at which margin is wiped out
        # For Exporter/IT: same concept from revenue side
        base_inr = deal_size * current_rate
        margin_amount = base_inr * (target_margin / 100)

        if business_type == 'Importer':
            # Rate must not rise above this for margin to survive
            break_even_rate = (base_inr + margin_amount) / deal_size
            be_risk = 'SAFE' if current_rate < break_even_rate else 'AT RISK'
        else:
            # Rate must not fall below this for margin to survive
            break_even_rate = (base_inr - margin_amount) / deal_size
            be_risk = 'SAFE' if current_rate > break_even_rate else 'AT RISK'

        # Exposure profile
        exposure = get_business_exposure(deal_size, business_type, current_rate)

        # Profit-at-Risk (use forecast bounds approximation)
        forecast_upper = predicted_rate * 1.02
        forecast_lower = predicted_rate * 0.98
        par = calculate_profit_at_risk(
            deal_size, business_type, current_rate, forecast_upper, forecast_lower
        )

        # Recommendation
        rec = get_recommendation(
            deal_size=deal_size,
            business_type=business_type,
            risk_score=risk_metrics['score'],
            risk_level=risk_metrics['level'],
            forecast_trend=trend,
            current_rate=current_rate,
            predicted_rate=predicted_rate
        )

        # Sensitivity matrix with margin_pct
        matrix = generate_sensitivity_matrix(deal_size, business_type, current_rate)
        for row in matrix:
            row['margin_pct'] = round(
                (row['gain_loss_inr'] / base_inr) * 100, 2
            ) if base_inr != 0 else 0

        return jsonify(sanitize({
            'current_rate': round(current_rate, 4),
            'break_even_rate': round(break_even_rate, 4),
            'break_even_risk': be_risk,
            'target_margin': target_margin,
            'exposure': exposure,
            'profit_at_risk': par,
            'recommendation': rec,
            'sensitivity_matrix': matrix,
            'risk_metrics': {
                'score': risk_metrics['score'],
                'level': risk_metrics['level'],
                'z_score': risk_metrics['z_score'],
                'is_anomaly': risk_metrics['is_anomaly'],
                'var_message': risk_metrics['var_message'],
                'inr_loss_amount': risk_metrics['inr_loss_amount'],
                '30d_volatility': risk_metrics['30d_volatility']
            }
        }))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# 4. Forecast Endpoints (Adwaitha's forecast_engine)
# ---------------------------------------------------------------------------
@app.route('/api/forecast', methods=['GET'])
def get_detailed_forecast():
    """
    Returns full Prophet forecast data for a specific currency.
    Query params:
      - currency: USD, GBP, EUR, JPY (default: USD)
      - days: forecast horizon (default: 7)
      - date: optional target date (YYYY-MM-DD).
              If PAST: uses data up to that date, forecasts 'days' from there.
              If FUTURE: forecasts far enough to cover date + days, then slices.
    """
    currency = request.args.get('currency', 'USD')
    days = int(request.args.get('days', 7))
    target_date = request.args.get('date', None)

    try:
        from data_engine import get_final_data
        import pandas as pd
        from datetime import datetime, timedelta

        df, _adf = get_final_data()
        last_data_date = df.index[-1]

        if target_date:
            cutoff = pd.to_datetime(target_date)

            if cutoff <= last_data_date:
                # PAST date: filter data up to that date, forecast 'days' forward
                df = df[df.index <= cutoff]
                if len(df) < 30:
                    return jsonify({
                        'status': 'error',
                        'message': f'Not enough data before {target_date}.'
                    }), 400
                result = run_forecast(currency=currency, days=days, df=df)
            else:
                # FUTURE date: forecast extra days so we can slice from target_date
                days_to_target = (cutoff - last_data_date).days
                total_days_needed = days_to_target + days
                result = run_forecast(currency=currency, days=total_days_needed, df=df)

                # Slice the forecast_table to only show 'days' entries starting from target_date
                if result.get('status') == 'success' and result.get('forecast_table'):
                    full_table = result['forecast_table']
                    sliced = [row for row in full_table if pd.to_datetime(row['ds']) >= cutoff]
                    sliced = sliced[:days]  # keep only 'days' worth

                    if sliced:
                        result['forecast_table'] = sliced
                        # Update the summary KPIs to match the sliced window
                        result['predicted_rate'] = round(sliced[-1]['yhat'], 4)
                        result['forecast_upper'] = round(sliced[-1]['yhat_upper'], 4)
                        result['forecast_lower'] = round(sliced[-1]['yhat_lower'], 4)
                        last_rate = result['current_rate']
                        new_pred = sliced[-1]['yhat']
                        result['change_percent'] = round(((new_pred - last_rate) / last_rate) * 100, 3)
                        result['trend'] = 'UP' if new_pred > last_rate else 'DOWN'
                        result['message'] = (
                            f"The {currency}/INR rate is currently {last_rate}. "
                            f"By {sliced[-1]['ds']} it is forecast to move to {round(new_pred, 4)} "
                            f"(trend: {result['trend']}). The 95% confidence band is "
                            f"{round(sliced[-1]['yhat_lower'], 4)} – {round(sliced[-1]['yhat_upper'], 4)}."
                        )
        else:
            result = run_forecast(currency=currency, days=days, df=df)

        return jsonify(sanitize(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/forecast/all', methods=['GET'])
def get_all_forecasts():
    """
    Returns forecasts for ALL 4 currencies in one call.
    Query params:
      - days: forecast horizon (default: 7)
    """
    days = int(request.args.get('days', 7))

    try:
        results = {}
        for cur in ['USD', 'GBP', 'EUR', 'JPY']:
            results[cur] = run_forecast(currency=cur, days=days)
        return jsonify(sanitize(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# 5. Plot Image Endpoints
# ---------------------------------------------------------------------------
@app.route('/api/plots', methods=['GET'])
def list_plots():
    """Lists all available plot images."""
    try:
        abs_plots = os.path.abspath(PLOTS_DIR)
        if not os.path.isdir(abs_plots):
            return jsonify({'plots': [], 'error': 'Plots directory not found'})

        files = [f for f in os.listdir(abs_plots)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))]
        return jsonify({'plots': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots/<path:filename>', methods=['GET'])
def serve_plot(filename):
    """Serves a specific plot image."""
    abs_plots = os.path.abspath(PLOTS_DIR)
    return send_from_directory(abs_plots, filename)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  FX API Bridge — Starting on http://localhost:5000")
    print("=" * 60)

    # Pre-load data
    if engine.run_preprocessing():
        print("[OK] Data engine initialized successfully.")
    else:
        print("[WARN] Data engine could not load. Some endpoints may fail.")

    app.run(host='0.0.0.0', port=5000, debug=True)
