import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Safety Stock Optimizer", layout="wide")

# --- TITLES AND TEXT ---
st.title("📦 Supply Chain: Safety Stock Optimizer")
st.markdown("""
This model links **Forecast Accuracy**, **Target Customer Service Level**, and **Safety Stock**. 
Adjust the parameters in the sidebar to see the impact on your required inventory.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Model Parameters")

# Updated max_value to 200000
avg_demand = st.sidebar.number_input("Average Period Demand (units)", min_value=10, max_value=200000, value=1000, step=10)
lead_time = st.sidebar.number_input("Lead Time (periods)", min_value=1, max_value=365, value=7, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Adjustable Levers")

# Sliders for the interactive parameters
forecast_accuracy = st.sidebar.slider("Forecast Accuracy (%)", min_value=50.0, max_value=99.0, value=80.0, step=1.0) / 100
service_level = st.sidebar.slider("Target Service Level (%)", min_value=80.0, max_value=99.9, value=95.0, step=0.1) / 100

# --- CALCULATIONS ---
# Calculate Z-score based on service level
z_score = norm.ppf(service_level)

# Approximate standard deviation of error based on forecast accuracy
sigma_d = avg_demand * (1 - forecast_accuracy)

# Calculate Safety Stock
safety_stock = z_score * sigma_d * np.sqrt(lead_time)

# --- RESULTS DISPLAY ---
col1, col2, col3 = st.columns(3)
col1.metric("Required Safety Stock", f"{int(safety_stock):,} units")
col2.metric("Target Service Level", f"{service_level*100:.1f}%")
col3.metric("Forecast Accuracy", f"{forecast_accuracy*100:.1f}%")

st.markdown("---")

# --- 3D VISUALIZATION ---
st.subheader("Impact Visualization")
st.markdown("This 3D surface shows how Safety Stock (Vertical Z-Axis) reacts to different combinations of Service Level and Forecast Accuracy.")

# Generate arrays for the axes
fa_range = np.linspace(0.50, 0.99, 50)
csl_range = np.linspace(0.80, 0.999, 50)

# Create a meshgrid
FA, CSL = np.meshgrid(fa_range, csl_range)

# Calculate Safety Stock for every point on the grid
Z_scores = norm.ppf(CSL)
Sigma_D = avg_demand * (1 - FA)
SS_Grid = Z_scores * Sigma_D * np.sqrt(lead_time)

# Plotly 3D Surface with updated hovertemplate
fig = go.Figure(data=[go.Surface(
    z=SS_Grid, 
    x=FA * 100, 
    y=CSL * 100,
    colorscale='Viridis',
    colorbar_title='Safety Stock',
    hovertemplate='<b>Forecast Accuracy:</b> %{x:.1f}%<br>' +
                  '<b>Service Level:</b> %{y:.1f}%<br>' +
                  '<b>Safety Stock:</b> %{z:,.0f} units<extra></extra>'
)])

# Updated dimensions (width and height)
fig.update_layout(
    title='Safety Stock vs. Service Level & Forecast Accuracy',
    scene=dict(
        xaxis_title='Forecast Accuracy (%)',
        yaxis_title='Service Level (%)',
        zaxis_title='Safety Stock (Units)'
    ),
    width=1200, 
    height=900, 
    margin=dict(l=65, r=50, b=65, t=90)
)

st.plotly_chart(fig, use_container_width=True)
