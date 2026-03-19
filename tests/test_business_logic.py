import pytest
import sys
import os

# Add backend to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from business_logic import get_recommendation, calculate_break_even_rate

def test_get_recommendation_critical_hedge():
    """Test that High risk and high score triggers a Critical Hedge warning."""
    # Action: get_recommendation(deal_size, business_type, risk_score, risk_level, forecast_trend, current_rate, predicted_rate)
    rec = get_recommendation(
        deal_size=100000, 
        business_type="Importer", 
        risk_score=85, 
        risk_level="High", 
        forecast_trend="UP", 
        current_rate=83.0, 
        predicted_rate=85.0
    )
    assert "HEDGE" in rec['action'].upper()
    assert rec['urgency'] == "IMMEDIATE"
    assert rec['hedge_percentage'] >= 75

def test_get_recommendation_wait():
    """Test that Low risk triggers a Wait action."""
    rec = get_recommendation(
        deal_size=100000, 
        business_type="Importer", 
        risk_score=20, 
        risk_level="Low", 
        forecast_trend="DOWN", 
        current_rate=83.0, 
        predicted_rate=81.0
    )
    assert "WAIT" in rec['action'].upper() or "SPOT" in rec['action'].upper()
    assert rec['hedge_percentage'] == 0

def test_calculate_break_even_importer():
    """Test break-even math for an Importer (cost increases are bad)."""
    rate = calculate_break_even_rate(100000, "Importer", 83.0, 5.0)
    # 5% target margin on 83.0 means they can absorb up to a 5% increase in cost
    expected_rate = 83.0 * (1 + 0.05)
    assert abs(rate['break_even_rate'] - expected_rate) < 0.01
    assert "Upper limit" in rate['warning']

def test_calculate_break_even_exporter():
    """Test break-even math for an Exporter (revenue drops are bad)."""
    rate = calculate_break_even_rate(100000, "Exporter", 83.0, 5.0)
    expected_rate = 83.0 * (1 - 0.05)
    assert abs(rate['break_even_rate'] - expected_rate) < 0.01
    assert "Lower limit" in rate['warning']
