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
    # Handle edge cases for Z-score
    sl = max(0.50, min(0.999, row['Targeted service level'] / 100))
    z = norm.ppf(sl)
    # Sigma (Demand Volatility) estimated from Forecast Accuracy
    sigma = row['Daily volume'] * (1 - (forecast_acc / 100))
    ss = z * sigma * np.sqrt(row['Lead-time to customer (days)'])
    return max(0, int(ss))

# --- TITLE ---
st.title("📦 Inventory Strategy & Sensitivity Dashboard")

# --- THEORY SECTION ---
with st.expander("📚 View Safety Stock Logic & Formula"):
    st.markdown(r"""
    The safety stock ($SS$) is calculated using the following relationship:
    $$SS = Z \times (Daily Volume \times (1 - Forecast Accuracy)) \times \sqrt{Lead Time}$$
    * **Z (Service Factor):** Derived from the Targeted Service Level.
    * **Forecast Error:** Represented as $(1 - Forecast Accuracy)$, acting as the multiplier for demand uncertainty.
    """)

# --- INITIAL DATA ---
initial_data = {
    "Category": ["Premium", "Premium", "Premium", "Mid-range", "Mid-range", "Budget", "Budget"],
    "Season": ["All Season", "Summer", "Winter", "All Season", "Summer", "All Season", "Winter"],
    "Lead-time to customer (days)": [15, 18, 14, 15, 18, 15, 14],
    "Daily volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000],
    "Targeted service level": [94.0, 95.0, 50.0, 85.0, 85.0, 80.0, 50.0]
}
df_init = pd.DataFrame(initial_data)

# --- SIDEBAR ---
st.sidebar.header("Global Levers")
global_fa = st.sidebar.slider("Current Forecast Accuracy (%)", 50.0, 99.0, 80.0, 0.5)
st.sidebar.info("The 3D Analysis on the right shows how the **Total Network SS** changes based on these global levers.")

# --- MAIN LAYOUT (Side-by-Side) ---
col_table, col_graph = st.columns([1, 1.2])

with col_table:
    st.subheader("1. Portfolio Table")
    # Editable Table
    edited_df = st.data_editor(
        df_init, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "Daily volume": st.column_config.NumberColumn(max_value=200000),
            "Targeted service level": st.column_config.NumberColumn(format="%.1f%%")
        }
    )
    
    # Calculate SS for each row
    edited_df['Safety Stock'] = edited_df.apply(lambda row: calculate_ss(row, global_fa), axis=1)
    
    # Summary Metrics
    total_volume = edited_df['Daily volume'].sum()
    total_ss = edited_df['Safety Stock'].sum()
    
    st.markdown("---")
    st.metric("Total Network Safety Stock", f"{total_ss:,} units")
    st.metric("Total Daily Portfolio Volume", f"{total_volume:,} units")
    
    st.download_button("Export to CSV", edited_df.to_csv(index=False), "inventory_plan.csv")

with col_graph:
    st.subheader("2. Total Network Sensitivity")
    
    # Sensitivity calculation based on TOTAL volume and average Lead Time
    avg_lt = edited_df['Lead-time to customer (days)'].mean()
    
    fa_range = np.linspace(50, 99, 40)
    sl_range = np.linspace(80, 99.9, 40)
    FA, SL = np.meshgrid(fa_range, sl_range)

    # 3D Calculation using TOTAL Volume
    Z_grid = norm.ppf(SL / 100)
    Sigma_total = total_volume * (1 - (FA / 100))
    SS_total_surface = Z_grid * Sigma_total * np.sqrt(avg_lt)

    # Create 3D Surface
    fig = go.Figure(data=[go.Surface(
        z=SS_total_surface, x=fa_range, y=sl_range,
        colorscale='Viridis',
        hovertemplate='<b>Forecast Accuracy:</b> %{x:.1f}%<br>' +
                      '<b>Service Level:</b> %{y:.1f}%<br>' +
                      '<b>Total Network SS:</b> %{z:,.0f} units<extra></extra>'
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title="Forecast Accuracy (%)",
            yaxis_title="Service Level (%)",
            zaxis_title="Total Safety Stock"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700 # Increased height for the column view
    )

    st.plotly_chart(fig, use_container_width=True)
