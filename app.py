import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Inventory Optimizer", layout="wide", initial_sidebar_state="expanded")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    csv_filename = 'InventoryModelInput.csv'
    df = None
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        # Fallback to empty template if no file uploaded yet
        df = pd.DataFrame(columns=['Category', 'Season', 'Lead-time to customer (days)', 'Daily volume', 'Targeted service level'])
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
        
        if pd.isna(sl_val) or pd.isna(daily_vol) or pd.isna(base_lt): return 0
        
        prob = sl_val / 100.0 if sl_val > 1.0 else sl_val
        prob = max(0.001, min(0.999, prob)) 
        
        z = norm.ppf(prob)
        adj_lt = base_lt * (1 + lt_adj_pct / 100)
        sigma = daily_vol * (1 - (forecast_acc / 100))
        
        ss = z * sigma * np.sqrt(max(0, adj_lt))
        return int(max(0, ss))
    except:
        return 0

# --- UI START ---
st.title("🎯 Network Inventory Strategy & Sensitivity")
st.markdown("Optimize your safety stock by balancing target service levels against demand and supply volatility.")

with st.expander("📝 Calculation Logic & Assumptions"):
    st.markdown(r"""
    **Formula:** $SS = Z \times (Daily Volume \times (1 - Forecast Accuracy)) \times \sqrt{Lead Time \times (1 + \Delta LT)}$
    * **Note:** Supply variability and Lead-Time variability are treated as deterministic (0 variance) in this specific model.
    """)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Global Levers")
    st.markdown("Apply network-wide adjustments here.")
    global_fa = st.slider("Current Forecast Accuracy (%)", 0.0, 100.0, 57.0, 0.5)
    lt_adj = st.slider("Lead-Time Adjustment (%)", -50, 100, 0, step=5, help="Simulate global supply chain delays or accelerations.")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload custom CSV", type="csv")

# Load Data
raw_df = pd.read_csv(uploaded_file) if uploaded_file else load_data()

if not raw_df.empty:
    def clean_sl(val):
        if isinstance(val, str): val = val.replace('%', '')
        v = pd.to_numeric(val, errors='coerce')
        if v is None: return 0.95
        return v / 100.0 if v > 1.0 else v

    if 'Targeted service level' in raw_df.columns:
        raw_df['sl_decimal'] = raw_df['Targeted service level'].apply(clean_sl)

    # Layout
    col_table, col_visuals = st.columns([1.3, 1], gap="large")

    with col_table:
        st.subheader("📋 SKU/Category Strategy")
        
        # Calculate Base Metrics First
        calc_df = raw_df.copy()
        calc_df['Safety Stock'] = calc_df.apply(lambda r: calculate_ss_safe(r, global_fa, lt_adj), axis=1)
        calc_df['sl_calc'] = calc_df['Targeted service level'].apply(clean_sl)
        
        total_ss = calc_df['Safety Stock'].sum()
        total_vol = pd.to_numeric(calc_df['Daily volume'], errors='coerce').sum()
        weighted_sl = (calc_df['sl_calc'] * pd.to_numeric(calc_df['Daily volume'], errors='coerce')).sum() / total_vol if total_vol > 0 else 0

        # High-Impact KPI Cards at the top
        m1, m2, m3 = st.columns(3)
        m1.metric("📦 Total Safety Stock", f"{int(total_ss):,} u")
        m2.metric("📈 Total Daily Volume", f"{int(total_vol):,} u")
        m3.metric("🎯 Wtd. Service Level", f"{weighted_sl:.1%}")
        st.markdown("")

        # Visual Data Editor
        table_height = min(1500, (len(raw_df) + 1) * 37)
        edited_df = st.data_editor(
            calc_df.drop(columns=['sl_decimal', 'sl_calc', 'Safety Stock']), 
            num_rows="dynamic", 
            use_container_width=True,
            height=table_height,
            column_config={
                "Targeted service level": st.column_config.ProgressColumn(
                    "Service Level Target",
                    help="Targeted probability of not stocking out",
                    format="%.2f",
                    min_value=0.5,
                    max_value=1.0,
                ),
                "Daily volume": st.column_config.NumberColumn("Daily Volume", format="%d units"),
                "Lead-time to customer (days)": st.column_config.NumberColumn("Lead Time (Days)", format="%d")
            }
        )
        
        # Recalculate based on edits for the visuals
        edited_df['Safety Stock'] = edited_df.apply(lambda r: calculate_ss_safe(r, global_fa, lt_adj), axis=1)

    with col_visuals:
        st.subheader("🌐 Network Sensitivity")
        
        # --- 3D SURFACE WITH CONTOURS ---
        fa_range = np.linspace(0, 100, 40)
        sl_range = np.linspace(0.80, 0.999, 40) 
        FA, SL = np.meshgrid(fa_range, sl_range)

        Z_grid = norm.ppf(SL)
        lts = pd.to_numeric(edited_df['Lead-time to customer (days)'], errors='coerce')
        avg_lt = lts.mean() * (1 + lt_adj / 100) if not lts.empty else 1
        
        Sigma_total = total_vol * (1 - (FA / 100))
        SS_surface = Z_grid * Sigma_total * np.sqrt(max(0.1, avg_lt))

        fig3d = go.Figure(data=[go.Surface(
            z=SS_surface, x=fa_range, y=sl_range * 100,
            colorscale='Turbo',
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)), # ADDED FLOOR CONTOUR
            hovertemplate='<b>FA:</b> %{x:.1f}%<br><b>Service:</b> %{y:.1f}%<br><b>Total SS:</b> %{z:,.0f}<extra></extra>'
        )])

        fig3d.update_layout(
            scene=dict(
                xaxis_title="Forecast Accuracy (%)", 
                yaxis_title="Service Level (%)", 
                zaxis_title="Safety Stock"
            ),
            margin=dict(l=0, r=0, b=0, t=10), height=500
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("---")

        # --- NEW 2D CATEGORY DRIVER CHART ---
        st.subheader("📊 Stock Drivers by Category")
        if 'Category' in edited_df.columns:
            # Group by category and sum the newly calculated safety stock
            cat_df = edited_df.groupby('Category')['Safety Stock'].sum().reset_index()
            cat_df = cat_df.sort_values('Safety Stock', ascending=True)
            
            fig_bar = px.bar(
                cat_df, x='Safety Stock', y='Category', orientation='h',
                text_auto='.2s', color='Safety Stock', color_continuous_scale='Turbo'
            )
            fig_bar.update_layout(
                xaxis_title="Safety Stock Units", yaxis_title="", 
                margin=dict(l=0, r=0, t=10, b=0), height=300,
                coloraxis_showscale=False # Hide colorbar to save space
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("Upload CSV or place 'InventoryModelInput.csv' in the directory to begin.")


