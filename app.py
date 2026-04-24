import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import os

# --- PAGE CONFIGURATION ---
# Fixed the typo here: set_page_config
st.set_page_config(page_title="Inventory Strategy Optimizer", layout="wide")

# --- DATA LOADING ---
def load_data():
    csv_filename = 'InventoryModelInput.csv'
    if os.path.exists(csv_filename):
        return pd.read_csv(csv_filename)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Inventory CSV", type="csv")
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
    return None

# --- CALCULATION FUNCTION ---
def calculate_ss_safe(row, forecast_acc, lt_adj_pct):
    try:
        # Targeted service level is treated as the final decimal (e.g. 0.95)
        sl_val = pd.to_numeric(row.get('Targeted service level', 0.95), errors='coerce')
        daily_vol = pd.to_numeric(row.get('Daily volume', 0), errors='coerce')
        base_lt = pd.to_numeric(row.get('Lead-time to customer (days)', 1), errors='coerce')
        
        if pd.isna(sl_val) or pd.isna(daily_vol) or pd.isna(base_lt):
            return 0
        
        # Clip to ensure value is within the 0-1 probability range for norm.ppf
        # This handles values like 0.95 correctly.
        sl = max(0.01, min(0.999, sl_val)) 
        z = norm.ppf(sl)
        
        adj_lt = base_lt * (1 + lt_adj_pct / 100)
        sigma = daily_vol * (1 - (forecast_acc / 100))
        
        ss = z * sigma * np.sqrt(max(0, adj_lt))
        return int(max(0, ss))
    except:
        return 0

# --- UI START ---
st.title("📦 Inventory Strategy & Sensitivity Dashboard")

with st.expander("📚 View Safety Stock Logic & Formula"):
    st.markdown(r"$$SS = Z \times (Daily Volume \times (1 - Forecast Accuracy)) \times \sqrt{Lead Time \times (1 + \Delta LT)}$$")

# Sidebar
st.sidebar.header("Global Levers")
global_fa = st.sidebar.slider("Current Forecast Accuracy (%)", 50.0, 99.0, 80.0, 0.5)
lt_adj = st.sidebar.slider("Lead-Time Adjustment (%)", -50, 100, 0, step=5)

raw_df = load_data()

if raw_df is not None:
    # CLEANING: If CSV has '94%', convert it to 0.94 once here.
    # This ensures 'sl_val' in the calculation function is always a decimal probability.
    if 'Targeted service level' in raw_df.columns and raw_df['Targeted service level'].dtype == 'object':
        raw_df['Targeted service level'] = raw_df['Targeted service level'].astype(str).str.replace('%', '', regex=False)
        raw_df['Targeted service level'] = pd.to_numeric(raw_df['Targeted service level']) / 100

    col_table, col_graph = st.columns([1, 1.2])

    with col_table:
        st.subheader("1. Portfolio Table")
        
        edited_df = st.data_editor(
            raw_df, 
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "Targeted service level": st.column_config.NumberColumn(
                    "Service Level",
                    help="Targeted service level as a decimal probability (e.g. 0.95)",
                    format="%.3f"
                )
            }
        )
        
        # Calculate Safety Stock
        edited_df['Safety Stock'] = edited_df.apply(
            lambda r: calculate_ss_safe(r, global_fa, lt_adj), axis=1
        )
        
        # Aggregates
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
        
        fa_range = np.linspace(50, 99, 40)
        sl_range = np.linspace(0.80, 0.999, 40) 
        FA, SL = np.meshgrid(fa_range, sl_range)

        Z_grid = norm.ppf(SL)
        Sigma_total = total_vol * (1 - (FA / 100))
        SS_surface = Z_grid * Sigma_total * np.sqrt(max(0.1, avg_lt))

        fig = go.Figure(data=[go.Surface(
            z=SS_surface, x=fa_range, y=sl_range * 100, 
            colorscale='Viridis',
            hovertemplate='<b>FA:</b> %{x:.1f}%<br><b>Service:</b> %{y:.1f}%<br><b>Total SS:</b> %{z:,.0f}<extra></extra>'
        )])

        fig.update_layout(
            scene=dict(xaxis_title="Forecast Accuracy %", yaxis_title="Service Level %", zaxis_title="Total SS"),
            margin=dict(l=0, r=0, b=0, t=30), height=750
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Awaiting data. Please check the CSV or upload via the sidebar.")


