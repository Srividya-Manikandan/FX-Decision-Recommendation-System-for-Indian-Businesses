import pytest
import sys
import os
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from forecast_engine import run_forecast
from risk_engine import calculate_risk_metrics

def test_prophet_forecast_bounds_logic():
    """
    REGRESSION: Ensure yhat bounds (confidence bands) are mathematically sound.
    yhat_lower MUST be <= yhat <= yhat_upper
    """
    # Create simple dummy dataset so Prophet doesn't fail
    dates = pd.date_range(start='1/1/2023', periods=100, freq='D')
    values = [80 + (i * 0.05) for i in range(100)] # steady trend
    df = pd.DataFrame({'Date': dates, 'USD': values})
    df.set_index('Date', inplace=True)
    
    # Need to suppress print statements or test the actual integration
    # The fix for the unicode error happened here, so we verify run_forecast works
    result = run_forecast(currency="USD", days=7, df=df)
    
    assert result['status'] == 'success'
    
    # Assert Bounds regression
    for day in result['forecast_table']:
        assert day['yhat_lower'] <= day['yhat']
        assert day['yhat'] <= day['yhat_upper']

def test_risk_metrics_anomaly_handling():
    """
    REGRESSION: Ensure Z-Score calculates correctly and triggers anomaly flags
    exactly when z >= 2 or z <= -2.
    """
    # Create dataset with a sudden massive spike (anomaly)
    dates = pd.date_range(start='1/1/2023', periods=60, freq='D')
    values = [80.0] * 59 + [95.0] # Massive spike on last day
    df = pd.DataFrame({'Date': dates, 'USD': values})
    df.set_index('Date', inplace=True)
    
    # Calculate returns and volatility as fx_engine would
    df['USD_Return'] = df['USD'].pct_change()
    df['USD_Volatility'] = df['USD_Return'].rolling(window=30).std()
    
    # The last day should be an anomaly
    metrics = calculate_risk_metrics(df, currency="USD", exposure_usd=100000)
    
    assert bool(metrics['is_anomaly']) is True
    assert abs(metrics['z_score']) > 2.0
    assert metrics['level'] == "High"
