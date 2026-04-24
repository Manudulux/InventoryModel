import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import os

# --- PAGE CONFIGURATION ---
st.set_config(page_title="Inventory Strategy Optimizer", layout="wide")

# --- DATA LOADING ---
def load_data():
    csv_filename = 'InventoryModelInput.csv'
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        return df
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Inventory CSV", type="csv")
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
    return None

# --- ROBUST CALCULATION FUNCTION ---
def calculate_ss_safe(row, forecast_acc, lt_adj_pct):
    try:
        # 1. Convert inputs to numeric
        # Targeted service level is used directly as a probability (e.g., 0.95)
        sl_val = pd.to_numeric(row.get('Targeted service level', 0.95), errors='coerce')
        daily_vol = pd.to_numeric(row.get('Daily volume', 0), errors='coerce')
        base_lt = pd.to_numeric(row.get('Lead-time to customer (days)', 1), errors='coerce')
        
        if pd.isna(sl_val) or pd.isna(daily_vol) or pd.isna(base_lt):
            return 0
        
        # 2. Math Logic 
        # sl is used directly as requested (no division by 100)
        # We clip to avoid Z-score infinity at exactly 1.0 or errors below 0.0
        sl = max(0.001, min(0.999, sl_val)) 
        z = norm.ppf(sl)
        
        # Lead Time adjustment
        adj_lt = base_lt * (1 + lt_adj_pct / 100)
        
        # Sigma (Demand Volatility) based on Forecast Accuracy
        sigma = daily_vol * (1 - (forecast_acc / 100))
        
        # Safety Stock formula
        ss = z * sigma * np.sqrt(max(0, adj_lt))
        return int(max(0, ss))
    except:
        return 0

# --- UI START ---
st.title("📦 Inventory Strategy & Sensitivity Dashboard")

with st.expander("📚 View Safety Stock Logic & Formula"):
    st.markdown(r"""
    The safety stock ($SS$) is calculated using the following relationship:
    $$SS = Z \times (Daily Volume \times (1 - Forecast Accuracy)) \times \sqrt{Lead Time \times (1 + \Delta LT)}$$
    * **$Z$ (Service Factor):** The inverse of the Normal Distribution for the **Targeted Service Level**.
    * **Forecast Error:** $(1 - Forecast Accuracy)$, the multiplier for demand uncertainty.
    * **$\Delta LT$:** Global percentage adjustment for Lead Times.
    """)

# Sidebar
st.sidebar.header("Global Levers")
global_fa = st.sidebar.slider("Current Forecast Accuracy (%)", 50.0, 99.0, 80.0, 0.5)
lt_adj = st.sidebar.slider("Lead-Time Adjustment (%)", -50, 100, 0, step=5)

raw_df = load_data()

if raw_df is not None:
    # Pre-clean CSV: if values are strings like '95%', convert to 0.95
    if 'Targeted service level' in raw_df.columns and raw_df['Targeted service level'].dtype == 'object':
        raw_df['Targeted service level'] = raw_df['Targeted service level'].str.replace('%', '', regex=False)
        raw_df['Targeted service level'] = pd.to_numeric(raw_df['Targeted service level']) / 100

    col_table, col_graph = st.columns([1, 1.2])

    with col_table:
        st.subheader("1. Portfolio Table")
        
        # Editable Table
        edited_df = st.data_editor(
            raw_df, 
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "Targeted service level": st.column_config.NumberColumn(
                    "Service Level",
                    help="Targeted service level as a decimal (e.g. 0.95)",
                    format="%.3f"
                )
            }
        )
        
        # Apply SS calculation
        edited_df['Safety Stock'] = edited_df.apply(
            lambda r: calculate_ss_safe(r, global_fa, lt_adj), axis=1
        )
        
        # Aggregate Metrics
        total_ss = edited_df['Safety Stock'].sum()
        total_vol = pd.to_numeric(edited_df['Daily volume'], errors='coerce').sum()
        
        lts = pd.to_numeric(edited_df['Lead-time to customer (days)'], errors='coerce')
        avg_lt = lts.mean() * (1 + lt_adj / 100) if not lts.empty else 1

        st.markdown("---")
        st.metric("Total Network Safety Stock", f"{int(total_ss):,} units")
        st.metric("Total Daily Portfolio Volume", f"{int(total_vol):,} units")
        st.download_button("Export Results", edited_df.to_csv(index=False), "inventory_results.csv")

    with col_graph:
        st.subheader("2. Total Network Sensitivity")
        
        # Grid for the 3D Surface
        fa_range = np.linspace(50, 99, 40)
        sl_range = np.linspace(0.80, 0.999, 40) # Service level as probability
        FA, SL = np.meshgrid(fa_range, sl_range)

        # 3D Calculation based on TOTAL volume
        Z_grid = norm.ppf(SL)
        Sigma_total = total_vol * (1 - (FA / 100))
        SS_surface = Z_grid * Sigma_total * np.sqrt(max(0.1, avg_lt))

        fig = go.Figure(data=[go.Surface(
            z=SS_surface, x=fa_range, y=sl_range * 100, # Display Y as % for readability
            colorscale='Viridis',
            hovertemplate='<b>FA:</b> %{x:.1f}%<br><b>Service:</b> %{y:.1f}%<br><b>Total SS:</b> %{z:,.0f}<extra></extra>'
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title="Forecast Accuracy (%)", 
                yaxis_title="Service Level (%)", 
                zaxis_title="Total Safety Stock"
            ),
            margin=dict(l=0, r=0, b=0, t=30), 
            height=750
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload your CSV file or ensure 'InventoryModelInput.csv' is in the root directory.")
