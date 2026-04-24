import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Inventory Strategy Optimizer", layout="wide")

# --- CUSTOM FUNCTIONS ---
def calculate_ss(row, forecast_acc):
    """Calculates Safety Stock for a single row."""
    # Z-score from Service Level
    z = norm.ppf(row['Targeted service level'] / 100)
    # Sigma (Demand Volatility) estimated from Forecast Accuracy
    sigma = row['Daily volume'] * (1 - (forecast_acc / 100))
    # SS = Z * Sigma * sqrt(LT)
    ss = z * sigma * np.sqrt(row['Lead-time to customer (days)'])
    return max(0, int(ss))

# --- TITLE & DESCRIPTION ---
st.title("📦 Multi-Category Safety Stock Optimizer")

# --- THEORY SECTION (How it's calculated) ---
with st.expander("📚 How is Safety Stock calculated? (The Formula)"):
    st.markdown(r"""
    The model uses the **Normal Distribution Lead-Time Demand** formula to link your parameters:
    
    $$SS = Z \times \sigma_{d} \times \sqrt{LT}$$
    
    **Where:**
    * **$Z$ (Service Factor):** The number of standard deviations required to meet your **Targeted Service Level**. (e.g., 95% service level $\approx$ 1.645).
    * **$\sigma_{d}$ (Demand Volatility):** Calculated here as $Daily Volume \times (1 - Forecast Accuracy)$. It represents the uncertainty in your demand.
    * **$LT$ (Lead Time):** The duration (in days) between placing an order and receiving it.
    
    **The Logic:** If you improve **Forecast Accuracy**, $\sigma_{d}$ drops, which linearly reduces Safety Stock. If you increase **Service Level**, $Z$ increases exponentially as you approach 100%, causing a sharp spike in inventory.
    """)

# --- INITIAL DATA (From your image) ---
initial_data = {
    "Category": ["Premium", "Premium", "Premium", "Mid-range", "Mid-range", "Budget", "Budget"],
    "Season": ["All Season", "Summer", "Winter", "All Season", "Summer", "All Season", "Winter"],
    "Lead-time to customer (days)": [15, 18, 14, 15, 18, 15, 14],
    "Daily volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000],
    "Targeted service level": [94.0, 95.0, 50.0, 85.0, 85.0, 80.0, 50.0]
}
df_init = pd.DataFrame(initial_data)

# --- USER INPUTS ---
st.sidebar.header("Global Levers")
# Global Forecast Accuracy affects the Sigma calculation for all rows
global_fa = st.sidebar.slider("Current Forecast Accuracy (%)", 50.0, 99.0, 80.0, 0.5)

st.subheader("1. Edit Your Product Categories")
st.info("Edit the 'Daily volume' or 'Targeted service level' directly in the table below.")
edited_df = st.data_editor(df_init, num_rows="dynamic", use_container_width=True)

# Perform Calculations
edited_df['Required Safety Stock'] = edited_df.apply(lambda row: calculate_ss(row, global_fa), axis=1)

# --- METRICS SUMMARY ---
st.markdown("---")
total_ss = edited_df['Required Safety Stock'].sum()
avg_sl = edited_df['Targeted service level'].mean()
st.metric("Total Network Safety Stock", f"{total_ss:,} units", help="Sum of safety stock across all rows above")

# --- 3D VISUALIZATION ---
st.subheader("2. Impact Visualization (Average Scenario)")

# Use averages from the table for the 3D surface context
avg_vol = edited_df['Daily volume'].mean()
avg_lt = edited_df['Lead-time to customer (days)'].mean()

fa_range = np.linspace(50, 99, 50)
sl_range = np.linspace(80, 99.9, 50)
FA, SL = np.meshgrid(fa_range, sl_range)

# Calculation for Surface
Z_grid = norm.ppf(SL / 100)
Sigma_grid = avg_vol * (1 - (FA / 100))
SS_surface = Z_grid * Sigma_grid * np.sqrt(avg_lt)

fig = go.Figure(data=[go.Surface(
    z=SS_surface, x=fa_range, y=sl_range,
    colorscale='Viridis',
    hovertemplate='<b>Forecast Accuracy:</b> %{x:.1f}%<br>' +
                  '<b>Service Level:</b> %{y:.1f}%<br>' +
                  '<b>Safety Stock:</b> %{z:,.0f} units<extra></extra>'
)])

fig.update_layout(
    title=f"Sensitivity Analysis (Based on Avg Volume: {int(avg_vol)})",
    scene=dict(
        xaxis_title="Forecast Accuracy (%)",
        yaxis_title="Service Level (%)",
        zaxis_title="Safety Stock"
    ),
    width=1200, height=800
)

st.plotly_chart(fig, use_container_width=True)

# --- DOWNLOAD DATA ---
st.download_button("Export Results to CSV", edited_df.to_csv(index=False), "inventory_plan.csv", "text/csv")

