import streamlit as st
import plotly.express as px
import pandas as pd

from src.synth_data import make_synthetic_transactions
from src.model_elasticity import derive_elasticity_cube
from src.uplift import compute_price_lift_impact

df = make_synthetic_transactions()

# 2️⃣ Build elasticity cube
cube = derive_elasticity_cube(df)

with st.sidebar:
    st.header("Controls")

    price_increase = st.slider(
        "Simulate Price Increase (%)",
        0.0, 5.0, 2.0, 0.5
    )

    st.subheader("Raise Score Weights")
    w_el = st.slider("Elasticity (reward)", 0.0, 1.0, 0.35, 0.05)
    w_mg = st.slider("Margin (reward)",     0.0, 1.0, 0.30, 0.05)
    w_rev = st.slider("Revenue uplift (reward)", 0.0, 1.0, 0.25, 0.05)
    w_risk = st.slider("Volume risk (penalty)", 0.0, 1.0, 0.10, 0.05)

    # Normalize reward weights so they sum to 1 (penalty stays separate)
    s = (w_el + w_mg + w_rev) or 1.0
    w_el, w_mg, w_rev = w_el / s, w_mg / s, w_rev / s

    st.subheader("Tier Thresholds")
    t1 = st.slider("Tier 1 threshold (Safe Raise)", 0.50, 0.90, 0.65, 0.01)
    t2 = st.slider("Tier 2 threshold (Test Raise)", 0.20, 0.80, 0.45, 0.01)

    if t2 >= t1:
        st.warning("Tier 2 threshold should be lower than Tier 1.")
sim_df = compute_price_lift_impact(
    cube,
    price_increase_pct=price_increase,
    w_elasticity=w_el,
    w_margin=w_mg,
    w_rev_uplift=w_rev,
    w_vol_risk=w_risk,
    t1=t1,
    t2=t2,
)


#st.sidebar.markdown("### Debug Info")
#st.sidebar.write("Price Std Dev:", df["net_price"].std())
#st.sidebar.write("Units Std Dev:", df["units"].std())

#st.write("Overall price-units correlation:",
#         df["net_price"].corr(df["units"]))

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue Lift",
            f"${sim_df['revenue_delta'].sum():,.0f}")
col2.metric("Avg Elasticity",
            f"{sim_df['elasticity'].mean():.2f}")
#col3.metric("Raise Candidates",
#            f"{(sim_df['elasticity'] > -1).sum()}")

col3.metric(
    "Tier 1 Safe Raises",
    f"{(sim_df['raise_tier'] == '🟢 Tier 1 – Safe Raise').sum():,}"
)

# Scatter Plot
#fig = px.scatter(
#    sim_df,
#    x="elasticity",
#    y="avg_margin",
#    color="segment",
#    hover_data=["sku","region"],
#    title="Elasticity vs Margin"
#)

#st.write("Current scoring config:", {
#    "price_increase_%": price_increase,
#    "weights": {"elasticity": w_el, "margin": w_mg, "rev_uplift": w_rev, "vol_risk_penalty": w_risk},
#    "thresholds": {"tier1": t1, "tier2": t2}
#})


fig_tier = px.histogram(
    sim_df,
    x="raise_tier",
    title="Raise Tier Distribution",
    color="raise_tier"
)
st.plotly_chart(fig_tier, use_container_width=True)


#st.plotly_chart(fig, use_container_width=True)

# Table
st.dataframe(
    sim_df.sort_values("revenue_delta", ascending=False)
    .head(20)
)
