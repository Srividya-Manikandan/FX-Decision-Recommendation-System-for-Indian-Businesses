# frontend_phases.md

# FX Recommendation System: Frontend Development Plan

This plan outlines the design and development of the **FX Decision Recommendation Dashboard**, inspired by high-performance financial analytics interfaces.

## Project Team
- **Srividya Manikandan**: FX Data Analysis & Preprocessing
- **Aadhithya Bharathi**: Business Exposure Modelling
- **Kanishkhan**: Risk Scoring Engine
- **Adwaitha**: Forecasting & Scenario Analysis
- **Adarsh**: Decision Recommendation & Integration (Lead)

---

## Phase 1: Design System & Infrastructure
- **Goal**: Establish the "Dark Premium" aesthetic (Glassmorphism, Vibrant Accents).
- **Tasks**:
    - Select typography (Inter/Outfit).
    - Define color palette: Primary (Electric Blue), Background (Deep Obsidian), Accents (Emerald Green for Low Risk, Crimson for High Risk).
    - Setup React/Next.js foundation with TailwindCSS for responsive layouts.

## Phase 2: Core Dashboard Layout
- **Goal**: Create the "Mission Control" view.
- **Components**:
    - **Header**: Project Title + Team Attribution.
    - **Sidebar**: Easy navigation between "Real-time Rates", "Risk Analysis", and "Recommendations".
    - **Main Area**: Responsive grid for charts and stats.

## Phase 3: Interactive Data Visualization
- **Goal**: Transform Srividya's and Adwaitha's data into interactive charts.
- **Tasks**:
    - Implement a customized Line Chart for USD-INR historical trends (using Chart.js or Recharts).
    - Add a "Volatility Heatmap" to visualize Risk over time.
    - Integration of "Forecasted Movement" overlays.

## Phase 4: Business Scenario & Decision Module
- **Goal**: Adarsh's Recommendation Logic & Aadhithya's Exposure Calculations.
- **Tasks**:
    - **Input Form**: Allow users to enter "USD Amount" and "Payment Deadline".
    - **Decision Card**: Large, high-impact card showing the generated recommendation (e.g., "CONVERT NOW").
    - **Exposure Impact**: Display the "Money at Stake" (Lakhs) calculation dynamically.

## Phase 5: Visual Assets & Polish (The "WOW" Factor)
- **Goal**: Add high-quality images and micro-animations.
- **Tasks**:
    - Generate futuristic currency/trading images using AI tools.
    - Add subtle hover effects and transition animations.
    - Final responsive testing across Mobile/Desktop.

---

## Verification Plan
- **UI Audit**: Ensure the design matches the "Bitcoin Analysis" reference in terms of premium feel.
- **Logic Sync**: Verify that the frontend recommendation matches the `Master_Notebook.ipynb` logic.
- **Image Check**: Ensure all media assets load correctly across devices.
