import pytest
import sys
import os
import json

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from api_bridge import app

@pytest.fixture
def client():
    # Setup flask test client
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_dashboard_endpoint_success(client):
    """Test that the main dashboard endpoint returns 200 and formatted JSON."""
    response = client.get('/api/dashboard?include_analysis=true&horizon=7')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'timestamp' in data
    assert 'pairs' in data
    
    # Check that USD is present and formatted
    assert 'USD' in data['pairs']
    usd_data = data['pairs']['USD']
    assert 'current_rate' in usd_data
    assert 'forecast_rate' in usd_data
    assert 'trend' in usd_data

def test_calculate_exposure_endpoint(client):
    """Test the POST /api/calculate-exposure endpoint logic."""
    payload = {
        "amount": 250000,
        "currency": "USD",
        "type": "Exporter"
    }
    response = client.post(
        '/api/calculate-exposure', 
        json=payload,
        content_type='application/json'
    )
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'current_rate' in data
    assert 'scenarios' in data
    assert 'sensitivity' in data
    assert len(data['scenarios']) > 0

def test_business_recommendation_endpoint(client):
    """Test the POST /api/business-recommendation endpoint orchestration."""
    payload = {
        "deal_size": 150000,
        "currency": "EUR",
        "type": "IT Services",
        "target_margin": 15.0
    }
    response = client.post(
        '/api/business-recommendation',
        json=payload
    )
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['currency'] == "EUR"
    assert 'exposure' in data
    assert 'profit_at_risk' in data
    assert 'recommendation' in data
    assert 'break_even' in data
    
    # Ensure numpy stripping worked (no NaN values crashing JSON)
    assert isinstance(data['recommendation']['urgency'], str)
