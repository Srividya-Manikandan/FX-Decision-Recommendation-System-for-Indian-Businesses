import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from exposure_engine import ExposureEngine

@pytest.fixture
def engine():
    return ExposureEngine()

def test_calculate_scenarios_importer(engine):
    """Ensure Importer loses money if currency appreciates (INR weakens)."""
    scenarios = engine.calculate_scenarios(amount=100000, current_rate=83.0, business_type="Importer")
    
    # Favorable scenario (For Importer, favorable is a drop in rate)
    bad_scenario = next(s for s in scenarios if s['scenario'] == 'Adverse')
    assert bad_scenario['gain_loss'] < 0  # Importer loses when rate goes up
    
    # Favorable scenario
    good_scenario = next(s for s in scenarios if s['scenario'] == 'Favorable')
    assert good_scenario['gain_loss'] > 0 # Importer gains when rate drops

def test_calculate_scenarios_exporter(engine):
    """Ensure Exporter gains money if currency appreciates (INR weakens)."""
    scenarios = engine.calculate_scenarios(amount=100000, current_rate=83.0, business_type="Exporter")
    
    # Favorable scenario (For Exporter, favorable is an increase in rate)
    good_scenario = next(s for s in scenarios if s['scenario'] == 'Favorable')
    assert good_scenario['gain_loss'] > 0  # Exporter gains when selling USD at higher rate
    
    # Adverse scenario
    bad_scenario = next(s for s in scenarios if s['scenario'] == 'Adverse')
    assert bad_scenario['gain_loss'] < 0 # Exporter loses when rate drops

def test_sensitivity_generation(engine):
    """Verify sensitivity matrix generates correctly with target margins."""
    res = engine.get_sensitivity(amount=100000, business_type="Importer")
    assert "priority" in res
    assert "zone" in res
    assert res['sensitivity_per_rupee'] == 100000
