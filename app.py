import streamlit as st
import plotly.express as px
import pandas as pd

from src.synth_data import make_synthetic_transactions
from src.model_elasticity import derive_elasticity_cube
from src.uplift import compute_price_lift_impact

st.set_page_config(layout="wide")
st.title("AI-Driven Pricing Intelligence Engine")
st.subheader("POC 1 – SKU × Segment × Region Elasticity")

# Load Data
df = make_synthetic_transactions()
cube = derive_elasticity_cube(df)

# Sidebar
price_increase = st.sidebar.slider(
    "Simulate Price Increase (%)", 0.0, 5.0, 2.0
)

sim_df = compute_price_lift_impact(cube, price_increase)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue Lift",
            f"${sim_df['revenue_delta'].sum():,.0f}")
col2.metric("Avg Elasticity",
            f"{sim_df['elasticity'].mean():.2f}")
col3.metric("Raise Candidates",
            f"{(sim_df['elasticity'] > -1).sum()}")

# Scatter Plot
fig = px.scatter(
    sim_df,
    x="elasticity",
    y="avg_margin",
    color="segment",
    hover_data=["sku","region"],
    title="Elasticity vs Margin"
)

st.plotly_chart(fig, use_container_width=True)

# Table
st.dataframe(
    sim_df.sort_values("revenue_delta", ascending=False)
    .head(20)
)
