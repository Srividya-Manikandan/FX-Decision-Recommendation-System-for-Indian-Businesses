# FX Recommendation System - Phase 1 Integration Complete

I have successfully implemented Adarsh's role as the **Integration Lead** for the project.

## Key Accomplishments

### 1. Master Integration Notebook
Created `adarsh_part/Master_Notebook.ipynb` which serves as the central hub for the team. 
- **Data Loading**: Automatically loads `cleaned_fx_data.csv` (Srividya's task).
- **Risk Evaluation**: Includes a placeholder function for Risk Scoring (Kanishkhan's task).
- **Forecasting**: Visualizes the last 30 days of USD-INR trends (Adwaitha's task).
- **Scenario Impact**: Calculates the financial impact of currency fluctuations in Lakhs (Aadhithya's task).

### 2. Decision Recommendation Engine
Developed the rule-based logic that provides actionable advice:
- **Condition**: High Risk + USD Strengthening $\rightarrow$ **Convert 100% Immediately**.
- **Condition**: Low Risk + Stable Rate $\rightarrow$ **Delay/Partial Conversion**.

## Verification Results
- [x] Notebook successfully imports existing preprocessed data.
- [x] Logic functions trigger correctly with sample inputs.
- [x] File structure follows the project requirements.

## Next Steps for the Team
- **Kanishkhan**: Replace the `get_risk_status` function with the final volatility-based logic.
- **Adwaitha**: Integrate the final Prophet/Moving Average prediction into the `predicted_trend` variable.
- **Aadhithya**: Use the `calculate_impact` function to show specific business losses in the final presentation.
