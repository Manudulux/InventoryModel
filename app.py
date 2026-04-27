import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Inventory Optimizer", layout="wide")

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
        df.columns = df.columns.str.strip()
    return df

# --- CALCULATION FUNCTION ---
def calculate_ss_safe(row, forecast_acc, lt_adj_pct):
    try:
        sl_raw = row.get('Targeted service level', 0.95)
        daily_vol = pd.to_numeric(row.get('Daily volume', 0), errors='coerce')
        base_lt = pd.to_numeric(row.get('Lead-time to customer (days)', 1), errors='coerce')
        
        if isinstance(sl_raw, str):
            sl_raw = sl_raw.replace('%', '')
        sl_val = pd.to_numeric(sl_raw, errors='coerce')
        
        if pd.isna(sl_val) or pd.isna(daily_vol) or pd.isna(base_lt):
            return 0
        
        # Treatment of SL as probability
        prob = sl_val / 100.0 if sl_val > 1.0 else sl_val
        prob = max(0.001, min(0.999, prob)) 
        
        z = norm.ppf(prob)
        adj_lt = base_lt * (1 + lt_adj_pct / 100)
        # Demand uncertainty based on Forecast Accuracy
        sigma = daily_vol * (1 - (forecast_acc / 100))
        
        ss = z * sigma * np.sqrt(max(0, adj_lt))
        return int(max(0, ss))
    except:
        return 0

# --- UI START ---
st.title("Sensitivity study of demand variability")

with st.expander("📝 Calculation Logic & Model Assumptions"):
    st.markdown(r"""
    ### **Safety Stock Formula**
    The model uses the standard demand-uncertainty formula:
    $$SS = Z \times (Daily Volume \times (1 - Forecast Accuracy)) \times \sqrt{Lead Time \times (1 + \Delta LT)}$$
    
    ### **Important Assumptions:**
    1. **Demand Variability Only:** This model specifically studies the impact of demand volatility.
    2. **Supply Variability:** **Not accounted for.** We assume supplier reliability is 100%.
    3. **Lead-Time Variability:** **Not accounted for.** Lead times are treated as deterministic (fixed), though they can be adjusted globally via the slider.
    4. **Normal Distribution:** We assume demand follows a normal distribution around the forecast.
    """)

# --- SIDEBAR ---
st.sidebar.header("Global Levers")
global_fa = st.sidebar.slider("Current Forecast Accuracy (%)", 0.0, 100.0, 57.0, 0.5)
lt_adj = st.sidebar.slider("Lead-Time Adjustment (%)", -50, 100, 0, step=5)

raw_df = load_data()

if raw_df is not None:
    # Pre-calculate probabilities for the Weighted Service Level metric
    def clean_sl(val):
        if isinstance(val, str): val = val.replace('%', '')
        v = pd.to_numeric(val, errors='coerce')
        if v is None: return 0.95
        return v / 100.0 if v > 1.0 else v

    if 'Targeted service level' in raw_df.columns:
        raw_df['sl_decimal'] = raw_df['Targeted service level'].apply(clean_sl)

    # Layout
    col_table, col_graph = st.columns([1.3, 1])

    with col_table:
        st.subheader("1. Portfolio Strategy")
        metric_container = st.container()
        
        # Display table (full height)
        table_height = min(2000, (len(raw_df) + 1) * 37)
        edited_df = st.data_editor(
            raw_df.drop(columns=['sl_decimal'] if 'sl_decimal' in raw_df.columns else []), 
            num_rows="dynamic", 
            use_container_width=True,
            height=table_height
        )
        
        # Re-run calculations on edited data
        edited_df['Safety Stock'] = edited_df.apply(
            lambda r: calculate_ss_safe(r, global_fa, lt_adj), axis=1
        )
        
        # Clean Service Level for Weighted Average
        edited_df['sl_calc'] = edited_df['Targeted service level'].apply(clean_sl)
        
        # Aggregates
        total_ss = edited_df['Safety Stock'].sum()
        total_vol = pd.to_numeric(edited_df['Daily volume'], errors='coerce').sum()
        
        # Weighted Average Service Level = Sum(Vol * SL) / Total Vol
        if total_vol > 0:
            weighted_sl = (edited_df['sl_calc'] * pd.to_numeric(edited_df['Daily volume'], errors='coerce')).sum() / total_vol
        else:
            weighted_sl = 0

        # Metrics at top
        with metric_container:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Safety Stock", f"{int(total_ss):,} units")
            m2.metric("Total Daily Volume", f"{int(total_vol):,} units")
            m3.metric("Weighted Avg Service Level", f"{weighted_sl:.1%}")
            st.markdown("---")

        st.download_button("Export Plan", edited_df.to_csv(index=False), "inventory_plan.csv")

    with col_graph:
        st.subheader("2. Network Sensitivity")
        fa_range = np.linspace(0, 100, 40)
        sl_range = np.linspace(0.80, 0.999, 40) 
        FA, SL = np.meshgrid(fa_range, sl_range)

        Z_grid = norm.ppf(SL)
        # Using lead time demand average for the graph
        lts = pd.to_numeric(edited_df['Lead-time to customer (days)'], errors='coerce')
        avg_lt = lts.mean() * (1 + lt_adj / 100) if not lts.empty else 1
        
        Sigma_total = total_vol * (1 - (FA / 100))
        SS_surface = Z_grid * Sigma_total * np.sqrt(max(0.1, avg_lt))

        fig = go.Figure(data=[go.Surface(
            z=SS_surface, x=fa_range, y=sl_range * 100,
            colorscale='Viridis',
            hovertemplate='<b>FA:</b> %{x:.1f}%<br><b>Service:</b> %{y:.1f}%<br><b>Total SS:</b> %{z:,.0f}<extra></extra>'
        )])

        fig.update_layout(
            scene=dict(xaxis_title="Forecast Accuracy %", yaxis_title="Service Level %", zaxis_title="SS"),
            margin=dict(l=0, r=0, b=0, t=30), height=700
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload CSV or place 'InventoryModelInput.csv' in the directory to begin.")
