from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from fx_engine import FXEngine
from exposure_engine import ExposureEngine
from business_logic import (
    get_business_exposure,
    calculate_profit_at_risk,
    get_recommendation as get_business_recommendation,
    generate_sensitivity_matrix,
)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5174", "http://localhost:5173"]}})

# Initialize Engines
engine = FXEngine()
exposure_engine = ExposureEngine()
engine.run_preprocessing()

@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns basic engine health and current spot rate."""
    # Optimization: If get_full_dashboard is too slow, we might want a lighter check here
    data = engine.get_full_dashboard()
    
    # Handle the new nested structure
    # Defaulting to USD for the general status check
    pairs = data.get('pairs', {})
    usd_data = pairs.get('USD', {})
    
    return jsonify({
        "status": "online",
        "current_rate": usd_data.get('current_rate'),
        "timestamp": data.get('timestamp')
    })

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Returns the complete dashboard dataset including analysis for a specific date."""
    target_date = request.args.get('date') # Expected format: YYYY-MM-DD
    # This triggers the forecast (Prophet) which we optimized to use only last 365 days
    data = engine.get_full_dashboard(include_analysis=True, target_date=target_date)
    return jsonify(data)

@app.route('/api/recommendation', methods=['GET'])
def get_recommendation():
    """Returns the final decision and risk metrics."""
    target_date = request.args.get('date')
    currency = request.args.get('currency', 'USD')
    
    data = engine.get_full_dashboard(target_date=target_date)
    
    pairs = data.get('pairs', {})
    curr_data = pairs.get(currency, {})
    
    if not curr_data:
         return jsonify({"error": f"No data for {currency}"}), 404
         
    return jsonify({
        "recommendation": curr_data.get('recommendation'),
        "risk_level": curr_data.get('risk_level'),
        "risk_score": curr_data.get('risk_score')
    })

@app.route('/api/analysis/correlations', methods=['GET'])
def get_correlations():
    """Specialized endpoint for the correlation heatmap."""
    target_date = request.args.get('date')
    data = engine.get_full_dashboard(include_analysis=True, target_date=target_date)
    correlations = data.get('analysis', {}).get('correlations', {})
    return jsonify(correlations)

@app.route('/api/calculate-exposure', methods=['POST'])
def calculate_exposure():
    """
    Calculates business exposure, scenarios, and sensitivity.
    Expects JSON: { "amount": 100000, "currency": "USD", "type": "Importer" }
    """
    try:
        req = request.json
        amount = float(req.get('amount', 0))
        currency = req.get('currency', 'USD')
        business_type = req.get('type', 'Importer')

        # Get latest rate from main engine
        # In a real scenario, we might want to check for the specific currency rate
        # For now, we assume USD/INR or fetch from engine
        dashboard_data = engine.get_full_dashboard()
        
        # If currency is not USD, we should ideally fetch that specific pair
        # For this prototype, we'll use the 'current_rate' which defaults to USD in many places 
        # or we fetch specifically:
        pairs = dashboard_data.get('pairs', {})
        pair_data = pairs.get(currency, {})
        current_rate = pair_data.get('current_rate', 83.0) # Fallback if missing

        if not current_rate:
             return jsonify({"error": f"Rate unavailable for {currency}"}), 400

        scenarios = exposure_engine.calculate_scenarios(amount, current_rate, business_type)
        sensitivity = exposure_engine.get_sensitivity(amount, business_type)

        return jsonify({
            "current_rate": current_rate,
            "scenarios": scenarios,
            "sensitivity": sensitivity
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/business-recommendation', methods=['POST'])
def business_recommendation():
    """
    Full business recommendation using business_logic.py.
    Expects JSON: { "deal_size": 100000, "currency": "USD", "type": "Importer" }
    Returns: exposure profile, profit-at-risk, recommendation, sensitivity matrix.
    """
    try:
        req = request.json
        deal_size = float(req.get('deal_size', 100000))
        currency = req.get('currency', 'USD')
        business_type = req.get('type', 'Importer')

        # Pull live data from fx_engine
        dashboard_data = engine.get_full_dashboard()
        pairs = dashboard_data.get('pairs', {})
        pair_data = pairs.get(currency, {})

        if not pair_data or 'error' in pair_data:
            return jsonify({"error": f"No data available for {currency}"}), 400

        current_rate = pair_data.get('current_rate', 85.0)
        predicted_rate = pair_data.get('forecast_7d', current_rate)
        risk_score = pair_data.get('risk_score', 50)
        risk_level = pair_data.get('risk_level', 'Medium')
        trend = pair_data.get('trend', 'UP')

        # Estimate forecast bounds (~1.5% band around prediction)
        band = current_rate * 0.015
        forecast_upper = predicted_rate + band
        forecast_lower = predicted_rate - band

        # Call business_logic functions
        exposure = get_business_exposure(deal_size, business_type, current_rate)
        par = calculate_profit_at_risk(
            deal_size, business_type, current_rate, forecast_upper, forecast_lower
        )
        recommendation = get_business_recommendation(
            deal_size, business_type, risk_score, risk_level, trend,
            current_rate, predicted_rate
        )
        sensitivity = generate_sensitivity_matrix(deal_size, business_type, current_rate)

        return jsonify({
            "currency": currency,
            "exposure": exposure,
            "profit_at_risk": par,
            "recommendation": recommendation,
            "sensitivity_matrix": sensitivity,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':

    print("FX Bridge API is starting on http://localhost:5000")
    # Using threaded=False to avoid Prophet issues on some systems if many requests hit at once
    app.run(host='0.0.0.0', port=5000, debug=True)
