import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import os

# --- PAGE CONFIGURATION ---
# Note: This must be the very first Streamlit command in the script.
st.set_page_config(page_title="Inventory Strategy Optimizer", layout="wide")

# --- DATA LOADING ---
def load_data():
    csv_filename = 'InventoryModelInput.csv'
    df = None
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Inventory CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    
    if df is not None:
        # FIX for "Always Zero": Strip whitespace from column names 
        # (e.g., 'Daily volume ' -> 'Daily volume')
        df.columns = df.columns.str.strip()
    return df

# --- ROBUST CALCULATION FUNCTION ---
def calculate_ss_safe(row, forecast_acc, lt_adj_pct):
    try:
        # Retrieve values with exact stripped keys
        sl_raw = row.get('Targeted service level', 0.95)
        daily_vol = pd.to_numeric(row.get('Daily volume', 0), errors='coerce')
        base_lt = pd.to_numeric(row.get('Lead-time to customer (days)', 1), errors='coerce')
        
        # Clean the service level value if it contains a % sign
        if isinstance(sl_raw, str):
            sl_raw = sl_raw.replace('%', '')
        sl_val = pd.to_numeric(sl_raw, errors='coerce')
        
        if pd.isna(sl_val) or pd.isna(daily_vol) or pd.isna(base_lt):
            return 0
        
        # LOGIC: "Do not divide by 100 again."
        # We treat sl_val as the probability. However, if the value is > 1.0 (e.g. 95), 
        # the norm.ppf math requires us to treat it as 0.95 to function.
        prob = sl_val
        if prob > 1.0:
            prob = prob / 100.0
            
        # Clamp probability for statistical stability (0.1% to 99.9%)
        prob = max(0.001, min(0.999, prob))
        z = norm.ppf(prob)
        
        # Adjusted Lead Time
        adj_lt = base_lt * (1 + lt_adj_pct / 100)
        
        # Sigma (Uncertainty) = Volume * Forecast Error (1 - FA)
        sigma = daily_vol * (1 - (forecast_acc / 100))
        
        ss = z * sigma * np.sqrt(max(0, adj_lt))
        return int(max(0, ss))
    except:
        return 0

# --- UI START ---
st.title("📦 Inventory Strategy & Sensitivity Dashboard")

with st.expander("📚 View Safety Stock Logic & Formula"):
    st.markdown(r"$$SS = Z \times (Daily Volume \times (1 - Forecast Accuracy)) \times \sqrt{Lead Time \times (1 + \Delta LT)}$$")

# --- SIDEBAR ---
st.sidebar.header("Global Levers")
# Updated default Forecast Accuracy to 57%
global_fa = st.sidebar.slider("Current Forecast Accuracy (%)", 0.0, 100.0, 57.0, 0.5)
lt_adj = st.sidebar.slider("Lead-Time Adjustment (%)", -50, 100, 0, step=5)

raw_df = load_data()

if raw_df is not None:
    # Minor pre-cleaning to ensure numbers in the editor
    if 'Targeted service level' in raw_df.columns:
        if raw_df['Targeted service level'].dtype == 'object':
            raw_df['Targeted service level'] = raw_df['Targeted service level'].astype(str).str.replace('%', '', regex=False)
            raw_df['Targeted service level'] = pd.to_numeric(raw_df['Targeted service level'], errors='coerce')

    col_table, col_graph = st.columns([1, 1.2])

    with col_table:
        st.subheader("1. Portfolio Table")
        edited_df = st.data_editor(raw_df, num_rows="dynamic", use_container_width=True)
        
        # Apply the calculation to every row
        edited_df['Safety Stock'] = edited_df.apply(
            lambda r: calculate_ss_safe(r, global_fa, lt_adj), axis=1
        )
        
        # Aggregate Summary Metrics
        total_ss = edited_df['Safety Stock'].sum()
        total_vol = pd.to_numeric(edited_df['Daily volume'], errors='coerce').sum()
        lts = pd.to_numeric(edited_df['Lead-time to customer (days)'], errors='coerce')
        avg_lt = lts.mean() * (1 + lt_adj / 100) if not lts.empty else 1

        st.markdown("---")
        st.metric("Total Network Safety Stock", f"{int(total_ss):,} units")
        st.metric("Total Daily Portfolio Volume", f"{int(total_vol):,} units")
        st.download_button("Export Results", edited_df.to_csv(index=False), "inventory_plan.csv")

    with col_graph:
        st.subheader("2. Total Network Sensitivity")
        
        # 3D Meshgrid
        fa_range = np.linspace(0, 100, 40)
        sl_range = np.linspace(0.80, 0.999, 40) 
        FA, SL = np.meshgrid(fa_range, sl_range)

        # 3D Surface Calculation
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
    st.info("Awaiting data. Please upload your CSV file via the sidebar or check its location in GitHub.")
