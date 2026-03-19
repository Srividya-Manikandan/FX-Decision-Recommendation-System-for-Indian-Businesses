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
            if "adarsh_part" in cwd or "notebooks" in cwd or "backend" in cwd:
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
            print("[INFO] Data Loaded Successfully from CSV.")
            print(f"[DEBUG] First Row: {df.iloc[0].to_dict()}")
            print(f"[DEBUG] Last Row: {df.iloc[-1].to_dict()}")
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

    def get_risk_assessment(self, currency='USD', exposure_usd=250000, target_date=None):
        """Calculates 60/40 weighted risk score for a target currency on a target date."""
        if self.df_master is None: return None
        
        # Determine the data slice
        if target_date:
            target_ts = pd.to_datetime(target_date)
            df_slice = self.df_master[self.df_master.index <= target_ts]
        else:
            df_slice = self.df_master

        if df_slice.empty: return {"error": f"Target date outside data range for {currency}"}
        
        vol_col = f'{currency}_Volatility'
        if vol_col not in self.df_master.columns:
            return {"error": f"Volatility data not found for {currency}"}

        # Volatility Score
        v_min, v_max = self.df_master[vol_col].min(), self.df_master[vol_col].max()
        latest_v = df_slice[vol_col].iloc[-1]
        
        # Avoid division by zero
        if v_max == v_min:
            vol_score = 0
        else:
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
            "status": "🚨 ACTION REQUIRED" if level == "High" else "Monitor",
            "date": df_slice.index[-1].strftime("%Y-%m-%d")
        }

    def get_forecast(self, currency='USD', days=7, show_plot=False, target_date=None):
        """Generates currency-specific forecast using Prophet for a target date."""
        if self.df_master is None: return None
        
        if target_date:
            target_ts = pd.to_datetime(target_date)
            df_slice = self.df_master[self.df_master.index <= target_ts]
        else:
            df_slice = self.df_master

        if df_slice.empty: return {"error": f"Target date outside data range for {currency}"}
        from forecast_engine import run_forecast
        
        # We pass df_slice directly to forecast_engine which handles data prep
        res = run_forecast(currency=currency, days=days, df=df_slice)
        
        if res.get("status") == "error":
            return {"error": res.get("message")}
            
        return res

    def get_historical_data(self, days=90):
        """Returns historical rates for the last N days."""
        if self.df_master is None: return None
        
        subset = self.df_master.tail(days).reset_index()
        subset['Date'] = subset['Date'].dt.strftime('%Y-%m-%d')
        return subset[['Date', 'USD', 'GBP', 'EUR', 'JPY']].to_dict(orient='records')

    def get_volatility_series(self, days=90):
        """Returns volatility trend for the last N days."""
        if self.df_master is None: return None
        
        subset = self.df_master.tail(days).reset_index()
        subset['Date'] = subset['Date'].dt.strftime('%Y-%m-%d')
        return subset[['Date', 'USD_Volatility']].to_dict(orient='records')

    def get_correlation_data(self, target_date=None):
        """Returns the correlation matrix for major currencies."""
        if self.df_master is None: return None
        
        # Slicing for point-in-time correlation
        if target_date:
            target_ts = pd.to_datetime(target_date)
            df_slice = self.df_master[self.df_master.index <= target_ts].tail(90)
        else:
            df_slice = self.df_master.tail(90)
            
        corr = df_slice[['USD', 'GBP', 'EUR', 'JPY']].corr()
        return corr.to_dict()

    def get_risk_map_data(self, target_date=None):
        """
        Generates data for 'Advanced Risk Mapping'.
        Calculates Volatility vs Market Sensitivity (USD Correlation).
        """
        if self.df_master is None: return None
        
        # Determine the data slice
        if target_date:
            target_ts = pd.to_datetime(target_date)
            df_slice = self.df_master[self.df_master.index <= target_ts]
        else:
            df_slice = self.df_master

        if df_slice.empty: return []

        results = []
        currencies = ['USD', 'GBP', 'EUR', 'JPY']
        
        # For USD correlation, we look at the last 90 days of the slice
        df_recent = df_slice.tail(90)
        correlations = df_recent[currencies].corr()['USD']
        
        for curr in currencies:
            vol = df_slice[f'{curr}_Volatility'].iloc[-1]
            corr_usd = correlations[curr]
            
            # Risk Score for bubble size: (Volatility * 0.7) + (Abs Corr * 0.3) normalized roughly
            risk_score = (vol * 1000) + (abs(corr_usd) * 10)
            
            results.append({
                "currency": curr,
                "volatility": round(vol, 6),
                "sensitivity": round(corr_usd, 4),
                "risk_score": round(risk_score, 2),
                "status": "Vulnerable" if vol > df_slice[f'{curr}_Volatility'].mean() * 1.5 else "Stable"
            })
            
        return results

    def get_recommendation(self, currency='USD', target_date=None, forecast_days=7):
        """Combines risk and forecast for final recommendation on target date for a specific currency."""
        risk = self.get_risk_assessment(currency=currency, target_date=target_date)
        forecast = self.get_forecast(currency=currency, days=forecast_days, target_date=target_date)
        
        if not risk or not forecast: return "Engine Not Ready"
        if "error" in risk: return risk["error"]
        if "error" in forecast: return forecast["error"]
        
        score = risk.get('score', 0)
        trend = forecast.get('trend', 'STABLE')
        
        if score > 75:
            return f"CRITICAL HEDGE ({currency}) - High volatility detected. Secure forward rates now."
        elif score > 55:
            if trend == "UP":
                return f"PARTIAL HEDGE (75%) ({currency}) - Rising rates and moderate risk. Act preemptively."
            else:
                return f"TACTICAL HEDGE (25%) ({currency}) - Volatility moderate, but trend is favorable."
        elif trend == "UP":
            return f"WATCHFUL BUYING ({currency}) - Low risk but rates are rising. Cover short-term needs."
        else:
            return f"WAIT / SPOT CONVERSION ({currency}) - Optimal conditions. No immediate hedging needed."

    def get_full_dashboard(self, show_plot=False, include_analysis=False, target_date=None, forecast_days=7):
        """Returns a consolidated summary of all metrics for all pairs for a given date."""
        self.run_preprocessing()
        if self.df_master is None: return {"error": "Data loading failed"}
        
        # Determine actual timestamp for display
        effective_date = target_date if target_date else self.df_master.index[-1].strftime("%Y-%m-%d")

        pairs_data = {}
        currencies = ['USD', 'GBP', 'EUR', 'JPY']
        
        for curr in currencies:
            risk = self.get_risk_assessment(currency=curr, target_date=effective_date)
            forecast = self.get_forecast(currency=curr, days=forecast_days, target_date=effective_date)
            
            if "error" in risk or "error" in forecast:
                pairs_data[curr] = {"error": risk.get("error") or forecast.get("error")}
                continue

            pairs_data[curr] = {
                "current_rate": forecast['current_rate'],
                "forecast_days": forecast_days,
                "forecast_rate": forecast['predicted_rate'],
                "forecast_lower": forecast.get('forecast_lower'),
                "forecast_upper": forecast.get('forecast_upper'),
                "forecast_table": forecast.get('forecast_table', []),
                "full_forecast": forecast.get('full_forecast', []),
                "trend": forecast['trend'],
                "risk_level": risk['level'],
                "risk_score": risk['score'],
                "recommendation": self.get_recommendation(currency=curr, target_date=effective_date, forecast_days=forecast_days)
            }

        data = {
            "timestamp": effective_date,
            "selected_date": effective_date,
            "pairs": pairs_data
        }

        if include_analysis:
            # For analysis, we slice up to the effective date
            # We use a slightly larger window check to ensure we have enough data for charts
            try:
                target_ts = pd.to_datetime(effective_date)
                df_slice = self.df_master[self.df_master.index <= target_ts]
                
                if df_slice.empty:
                    # Fallback to the very first available data point if requested date is too early
                    df_slice = self.df_master.head(90)
                
                data["analysis"] = {
                "historical_90d": self._get_slice_historical(df_slice, 90),
                "volatility_trend": self._get_slice_volatility(df_slice, 90),
                "correlations": self.get_correlation_data(target_date=effective_date),
                "risk_map": self.get_risk_map_data(target_date=effective_date)
            }
            except Exception as e:
                print(f"Analysis error: {e}")
                data["analysis"] = {"error": str(e)}
            
        return data

    def _get_slice_historical(self, df, days):
        df_sub = df.tail(days).reset_index()
        results = []
        for _, row in df_sub.iterrows():
            results.append({
                "Date": row['Date'].strftime("%Y-%m-%d"),
                "USD": round(row['USD'], 4),
                "GBP": round(row['GBP'], 4),
                "EUR": round(row['EUR'], 4),
                "JPY": round(row['JPY'], 4)
            })
        return results

    def _get_slice_volatility(self, df, days):
        df_sub = df.tail(days).reset_index()
        results = []
        for _, row in df_sub.iterrows():
            def safe_val(val):
                try:
                    v = float(val)
                    return round(v, 6) if not pd.isna(v) else 0
                except:
                    return 0

            results.append({
                "Date": row['Date'].strftime("%Y-%m-%d"),
                "USD_Volatility": safe_val(row.get('USD_Volatility', 0)),
                "GBP_Volatility": safe_val(row.get('GBP_Volatility', 0)),
                "EUR_Volatility": safe_val(row.get('EUR_Volatility', 0)),
                "JPY_Volatility": safe_val(row.get('JPY_Volatility', 0))
            })
        return results

    def _get_slice_correlations(self, df):
        corr = df[['USD', 'GBP', 'EUR', 'JPY']].tail(100).corr()
        return corr.to_dict()

# For Testing
if __name__ == "__main__":
    engine = FXEngine()
    if engine.run_preprocessing():
        # Enabled plotting for standalone run
        data = engine.get_full_dashboard(show_plot=True)
        
        # Extract USD data for the print summary
        usd_data = data['pairs'].get('USD', {})
        
        print("\n" + "="*50)
        print("     FX DECISION RECOMMENDATION ENGINE")
        print("="*50)
        print(f"Timestamp:          {data['timestamp']}")
        print(f"Current Spot Rate:  Rs. {usd_data.get('current_rate', 'N/A')}")
        print(f"7-Day Forecast:     Rs. {usd_data.get('forecast_7d', 'N/A')}")
        print(f"Movement Trend:     {usd_data.get('trend', 'N/A')}")
        print(f"Risk Assessment:    {usd_data.get('risk_level', 'N/A').upper()}")
        print(f"Risk Score:         {usd_data.get('risk_score', 'N/A')}")
        print("-"*50)
        print(f"FINAL RECOMMENDATION: \n{usd_data.get('recommendation', 'N/A')}")
        print("="*50 + "\n")
    else:
        print(" Error: Could not initialize engine. Check if data/raw/ files exist.")
