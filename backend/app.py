import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# Import analytical engines
from data_engine import get_final_data
from forecast_engine import run_forecast
from risk_engine import calculate_risk_metrics
from business_logic import (
    get_business_exposure,
    calculate_profit_at_risk,
    get_recommendation as get_business_recommendation,
    generate_sensitivity_matrix,
    calculate_break_even_rate
)

# ---------------------------------------------------------------------------
# 1. Page Configuration (Bloomberg Style)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FX Insight | Decision Support System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark, professional theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4451;
    }
    .recommendation-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
        border-left: 5px solid;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #3e4451;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Sidebar Implementation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ FX Decision System")
    st.info("Capstone Project: 23CSE452")
    
    st.subheader("👥 Project Team")
    st.markdown("""
    - **Srividya M.** (Data Pipeline)
    - **Aadhithya B.** (Exposure Logic)
    - **Kanishkhan** (Risk Engine)
    - **Adwaitha** (Forecasting)
    - **Adarsh** (Integration Lead)
    """)
    
    st.divider()
    
    st.subheader("⚙️ Analysis Parameters")
    selected_currency = st.selectbox("Currency Pair", ["USD", "EUR", "GBP", "JPY"], index=0)
    deal_size = st.number_input("Deal Size (Foreign Currency)", min_value=1000, value=100000, step=5000)
    business_type = st.radio("Business Position", ["Importer", "Exporter", "IT Services"])
    
    # Pass the actual business type to the engines
    # (Internally, they handle anything non-Importer as Exporter logic)

# ---------------------------------------------------------------------------
# 3. Data Loading & Engine Synchronization
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_and_sync():
    df, adf_results = get_final_data()
    return df, adf_results

with st.spinner("Synchronizing Market Data..."):
    df_master, adf = load_and_sync()

# ---------------------------------------------------------------------------
# 4. Header Section (Live Metrics)
# ---------------------------------------------------------------------------
st.title("💎 Global FX Command Center")
latest_date = df_master.index[-1].strftime("%A, %d %B %Y")
st.caption(f"Market Status: Online | Last Update: {latest_date}")

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

# Get current and previous rates for the selected currency
current_rate = df_master[selected_currency].iloc[-1]
prev_rate = df_master[selected_currency].iloc[-2]
change = current_rate - prev_rate
change_pct = (change / prev_rate) * 100

# Get forecast data
forecast_res = run_forecast(currency=selected_currency, df=df_master)
# Get risk data
risk_metrics = calculate_risk_metrics(df_master, selected_currency, exposure_usd=deal_size)

with col1:
    st.metric(f"Live {selected_currency}/INR", f"₹{current_rate:.4f}", f"{change_pct:+.2f}%")

with col2:
    trend_arrow = "↗️" if forecast_res['trend'] == "UP" else "↘️"
    st.metric("7-Day Forecast", f"₹{forecast_res['predicted_rate']:.2f}", f"{trend_arrow} {forecast_res['trend']}")

with col3:
    risk_level = risk_metrics['level']
    st.metric("Risk Assessment", f"{risk_metrics['score']}/100", risk_level, delta_color="inverse")

with col4:
    anom_status = "⚠️ ANOMALY" if risk_metrics['is_anomaly'] else "✅ STABLE"
    st.metric("Market Volatility", f"{risk_metrics['30d_volatility']:.4f}", anom_status)

st.divider()

# ---------------------------------------------------------------------------
# 5. Strategic Recommendation Logic
# ---------------------------------------------------------------------------
rec = get_recommendation(
    deal_size=deal_size,
    business_type=business_type,
    risk_score=risk_metrics['score'],
    risk_level=risk_metrics['level'],
    forecast_trend=forecast_res['trend'],
    current_rate=current_rate,
    predicted_rate=forecast_res['predicted_rate']
)

# Visual Box for recommendation
bg_color = {"red": "#721c24", "orange": "#856404", "yellow": "#856404", "blue": "#0c5460", "green": "#155724"}.get(rec['color'], "#333")
border_color = {"red": "#f8d7da", "orange": "#ffeeba", "yellow": "#ffeeba", "blue": "#bee5eb", "green": "#c3e6cb"}.get(rec['color'], "#666")

st.subheader("💡 Strategic Recommendation")
st.markdown(f"""
<div class="recommendation-box" style="background-color: {bg_color}; border-color: {border_color};">
    <h2 style="margin:0; color: white;">{rec['action']}</h2>
    <p style="font-size: 1.1em; color: {border_color};"><b>URGENCY: {rec['urgency']} | HEDGE TARGET: {rec['hedge_percentage']}%</b></p>
    <p style="color: white;">{rec['reasoning']}</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 6. Interactive Charts & Analysis
# ---------------------------------------------------------------------------
chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 Forecasting & Trends", "⚖️ Exposure & Sensitivity", "🛡️ Quant Dashboard"])

with chart_tab1:
    col_c1, col_c2 = st.columns([2, 1])
    
    with col_c1:
        st.subheader(f"{selected_currency}/INR Forecast (Prophet Model)")
        # Plotly chart for forecast
        f_df = pd.DataFrame(forecast_res['full_forecast'])
        f_df['ds'] = pd.to_datetime(f_df['ds'])
        
        # Historical portion (last 90 days)
        hist_df = df_master[[selected_currency]].tail(90).reset_index()
        hist_df.columns = ['Date', 'Value']
        
        fig = go.Figure()
        # Historical Trace
        fig.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Value'], name="Historical", line=dict(color='#3498db', width=3)))
        
        # Forecast Trace
        future_df = pd.DataFrame(forecast_res['forecast_table'])
        future_df['ds'] = pd.to_datetime(future_df['ds'])
        
        # Add confidence band
        fig.add_trace(go.Scatter(
            x=list(future_df['ds']) + list(future_df['ds'])[::-1],
            y=list(future_df['yhat_upper']) + list(future_df['yhat_lower'])[::-1],
            fill='toself',
            fillcolor='rgba(46, 204, 113, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence Band'
        ))
        
        fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat'], name="Forecast", line=dict(color='#2ecc71', width=3, dash='dash')))
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title=f"INR per {selected_currency}",
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_c2:
        st.subheader("Expected Movement")
        st.write(forecast_res['message'])
        
        move_pct = forecast_res['change_percent']
        st.metric("Predicted Change", f"{move_pct:+.2f}%", f"{rec['potential_impact_inr']:,.0f} INR Delta")
        
        with st.expander("Model Stationarity Data (ADF Test)"):
            curr_adf = adf.get(selected_currency, {})
            st.json(curr_adf)

with chart_tab2:
    st.subheader("Business Impact Analysis")
    
    # Exposure Profile
    profile = get_business_exposure(deal_size, business_type, current_rate)
    par = calculate_profit_at_risk(deal_size, business_type, current_rate, forecast_res['forecast_upper'], forecast_res['forecast_lower'])
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.write(f"**Current Base Value:** ₹{profile['base_inr_value']:,.2f}")
        st.write(f"**Risk Zone:** :{profile['risk_zone'].lower()}[{profile['risk_zone']}]")
        st.write(f"**Exposure Message:** {profile['message']}")
    
    with col_p2:
        st.write(f"**Worst Case Impact:** ₹{par['profit_at_risk_inr']:,.0f}")
        st.write(f"**Maximum Profit-at-Risk:** {par['profit_at_risk_lakhs']} Lakhs")
        st.warning(par['warning'])

    # Sensitivity Heatmap
    st.subheader("What-If Matrix (Sensitivity)")
    matrix_data = generate_sensitivity_matrix(deal_size, business_type, current_rate)
    matrix_df = pd.DataFrame(matrix_data)
    
    # Format for display
    display_df = matrix_df[['scenario', 'new_rate', 'gain_loss_inr', 'impact_label']].copy()
    display_df.columns = ['Scenario', 'Rate (₹)', 'Impact (₹)', 'Status']
    
    def color_impact(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'

    st.table(display_df)

with chart_tab3:
    st.subheader("Technical Risk Metrics")
    col_r1, col_r2 = st.columns([1, 1])
    
    with col_r1:
        # Gauge for Risk Score
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_metrics['score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Composite Risk Index", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "white"},
                'bgcolor': "black",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': 'green'},
                    {'range': [40, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_metrics['score']
                }
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_r2:
        st.subheader("Value-at-Risk (VaR)")
        st.info(risk_metrics['var_message'])
        
        st.subheader("Anomaly Detection (Z-Score)")
        st.write(f"The current rate is **{abs(risk_metrics['z_score'])}** standard deviations away from the 30-day mean.")
        if risk_metrics['is_anomaly']:
            st.error("Market anomaly detected. Standard pricing models may be unreliable.")
        else:
            st.success("Price action is within normal statistical bounds.")

# ---------------------------------------------------------------------------
# 7. Reporting Section
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📝 Export Executive Report")
if st.button("Generate PDF Summary"):
    st.write("Generating report... (This would normally trigger a PDF generator like FPDF)")
    st.download_button(
        label="Download Analysis Data (CSV)",
        data=df_master.tail(30).to_csv(),
        file_name=f"FX_Report_{selected_currency}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

