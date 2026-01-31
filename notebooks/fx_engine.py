import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime

class FXEngine:
    """
    Consolidated FX Decision Recommendation Engine.
    Handles data preprocessing, exposure modeling, risk scoring, and forecasting.
    """

    def __init__(self, base_dir=None):
        if base_dir is None:
            cwd = os.getcwd()
            if "adarsh_part" in cwd or "notebooks" in cwd:
                self.base_dir = os.path.abspath(os.path.join(cwd, ".."))
            else:
                self.base_dir = cwd
        else:
            self.base_dir = base_dir
        
        self.raw_rbi = os.path.join(self.base_dir, 'data', 'raw', 'RBI_BankWise(2k16-26).xlsx')
        self.raw_fbil = os.path.join(self.base_dir, 'data', 'raw', 'Reference_Rates.xlsx')
        self.processed_path = os.path.join(self.base_dir, 'data', 'processed', 'cleaned_fx_data.csv')
        self.df_master = None

    def run_preprocessing(self):
        """Merges RBI and FBIL data and calculates volatility."""
        if os.path.exists(self.processed_path):
            df = pd.read_csv(self.processed_path)
            if 'Unnamed: 0' in df.columns:
                df = df.rename(columns={'Unnamed: 0': 'Date'})
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            self.df_master = df
            return True

        if not os.path.exists(self.raw_rbi) or not os.path.exists(self.raw_fbil):
            return False

        # Preprocessing logic
        df_rbi = pd.read_excel(self.raw_rbi)
        df_rbi['Date'] = pd.to_datetime(df_rbi['Date'], dayfirst=True)
        df_rbi = df_rbi.rename(columns={'USD (INR / 1 USD)': 'USD', 'GBP (INR / 1 GBP)': 'GBP', 'EUR (INR / 1 EUR)': 'EUR', 'JPY (INR / 100 JPY)': 'JPY'})
        df_rbi = df_rbi[['Date', 'USD', 'GBP', 'EUR', 'JPY']].set_index('Date')

        df_fbil = pd.read_excel(self.raw_fbil, skiprows=2)
        df_fbil['Date'] = pd.to_datetime(df_fbil['Date'], dayfirst=True)
        mapping = {'INR / 1 USD': 'USD', 'INR/1 USD': 'USD', 'INR / 1 GBP': 'GBP', 'INR / 1 EUR': 'EUR', 'INR / 100 JPY': 'JPY'}
        df_fbil['Currency'] = df_fbil['Currency Pairs'].map(mapping)
        df_fbil = df_fbil.dropna(subset=['Currency'])
        df_fbil_pivot = df_fbil.pivot_table(index='Date', columns='Currency', values='Rate')

        df_combined = df_rbi.combine_first(df_fbil_pivot).sort_index()
        full_range = pd.date_range(start=df_combined.index.min(), end=df_combined.index.max(), freq='D')
        df_final = df_combined.reindex(full_range).ffill()

        for cur in ['USD', 'GBP', 'EUR', 'JPY']:
            df_final[f'{cur}_Return'] = df_final[cur].pct_change()
            df_final[f'{cur}_Volatility'] = df_final[f'{cur}_Return'].rolling(window=30).std()

        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        df_final.to_csv(self.processed_path)
        self.df_master = df_final
        return True

    def get_exposure_impact(self, usd_amount=100000, change_percent=1.0):
        """Calculates financial impact for different business types."""
        if self.df_master is None: return None
        
        latest_usd = self.df_master['USD'].iloc[-1]
        changed_rate = latest_usd * (1 + change_percent/100)
        diff = changed_rate - latest_usd
        
        return [
            {"type": "Importer", "usd_exposure": usd_amount, "impact_inr": usd_amount * diff, "description": "Cost Increase"},
            {"type": "Exporter", "usd_exposure": usd_amount, "impact_inr": usd_amount * -diff, "description": "Revenue Decrease"},
            {"type": "IT Services", "usd_exposure": usd_amount, "impact_inr": usd_amount * diff, "description": "Revenue Change"}
        ]

    def get_risk_assessment(self, exposure_usd=250000):
        """Calculates 60/40 weighted risk score."""
        if self.df_master is None: return None
        
        # Volatility Score
        v_min, v_max = self.df_master['USD_Volatility'].min(), self.df_master['USD_Volatility'].max()
        latest_v = self.df_master['USD_Volatility'].iloc[-1]
        vol_score = ((latest_v - v_min) / (v_max - v_min)) * 100
        
        # Exposure Score (Normalized to 500k scale)
        exp_score = ((exposure_usd - 50000) / (500000 - 50000)) * 100
        exp_score = max(0, min(100, exp_score))
        
        final_score = (0.6 * vol_score) + (0.4 * exp_score)
        level = "Low" if final_score < 40 else "Medium" if final_score <= 70 else "High"
        
        return {
            "score": round(final_score, 2),
            "level": level,
            "volatility": round(latest_v, 6),
            "status": "🚨 ACTION REQUIRED" if level == "High" else "Monitor"
        }

    def get_forecast(self, days=7, show_plot=False):
        """Generates USD-INR forecast using Prophet."""
        if self.df_master is None: return None
        
        pdf = self.df_master.reset_index()[['Date', 'USD']]
        pdf.columns = ['ds', 'y']
        
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(pdf)
        
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        
        if show_plot:
            fig = m.plot(forecast)
            plt.title("USD-INR Exchange Rate Forecast (7 Days)")
            plt.xlabel("Date")
            plt.ylabel("Rate (INR/USD)")
            plt.show()

        latest_pred = forecast['yhat'].iloc[-1]
        return {
            "predicted_rate": round(latest_pred, 4),
            "current_rate": round(self.df_master['USD'].iloc[-1], 4),
            "trend": "UP" if latest_pred > self.df_master['USD'].iloc[-1] else "DOWN"
        }

    def get_recommendation(self):
        """Combines risk and forecast for final recommendation."""
        risk = self.get_risk_assessment()
        forecast = self.get_forecast()
        
        if not risk or not forecast: return "Engine Not Ready"
        
        if risk['level'] == "High":
            return "HEDGE IMMEDIATELY - Forward Contract Recommended"
        elif forecast['trend'] == "UP" and risk['level'] != "Low":
            return "CONVERT 50% - Partial Hedge due to rising trend"
        else:
            return "WAIT / SPOT CONVERSION - No immediate risk"

    def get_full_dashboard(self, show_plot=False):
        """Returns a consolidated summary of all metrics."""
        self.run_preprocessing()
        if self.df_master is None: return {"error": "Data loading failed"}
        
        risk = self.get_risk_assessment()
        forecast = self.get_forecast(show_plot=show_plot)
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_rate": round(self.df_master['USD'].iloc[-1], 4),
            "forecast_7d": forecast['predicted_rate'],
            "trend": forecast['trend'],
            "risk_level": risk['level'],
            "risk_score": risk['score'],
            "recommendation": self.get_recommendation()
        }

# For Testing
if __name__ == "__main__":
    engine = FXEngine()
    if engine.run_preprocessing():
        # Enabled plotting for standalone run
        data = engine.get_full_dashboard(show_plot=True)
        
        print("\n" + "="*50)
        print("     FX DECISION RECOMMENDATION ENGINE")
        print("="*50)
        print(f"Timestamp:          {data['timestamp']}")
        print(f"Current Spot Rate:  Rs. {data['current_rate']:.4f}")
        print(f"7-Day Forecast:     Rs. {data['forecast_7d']:.4f}")
        print(f"Movement Trend:     {data['trend']}")
        print(f"Risk Assessment:    {data['risk_level'].upper()}")
        print(f"Risk Score:         {data['risk_score']}")
        print("-"*50)
        print(f"FINAL RECOMMENDATION: \n{data['recommendation']}")
        print("="*50 + "\n")
    else:
        print(" Error: Could not initialize engine. Check if data/raw/ files exist.")
