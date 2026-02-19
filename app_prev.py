import streamlit as st
import pandas as pd
import plotly.express as px

from src.synth_data import make_synthetic_transactions
from src.model_elasticity import derive_elasticity_cube
from src.uplift import compute_price_lift_impact


st.set_page_config(page_title="Pricing Intelligence Engine – POC1", layout="wide")

st.title("AI-Driven Pricing Intelligence Engine")
st.caption("POC 1 — SKU × Segment × Region Elasticity + Raise Score (Synthetic data)")

# -----------------------------
# Sidebar: Controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    price_increase = st.slider(
        "Simulate Price Increase (%)",
        0.0, 5.0, 2.0, 0.5
    )

    st.subheader("Raise Score Weights")
    w_el = st.slider("Elasticity (reward)", 0.0, 1.0, 0.35, 0.05)
    w_mg = st.slider("Margin (reward)", 0.0, 1.0, 0.30, 0.05)
    w_rev = st.slider("Revenue uplift (reward)", 0.0, 1.0, 0.25, 0.05)
    w_risk = st.slider("Volume risk (penalty)", 0.0, 1.0, 0.10, 0.05)

    # Normalize reward weights to sum to 1 (penalty stays separate)
    s = (w_el + w_mg + w_rev) or 1.0
    w_el, w_mg, w_rev = w_el / s, w_mg / s, w_rev / s

    st.subheader("Tier Thresholds")
    t1 = st.slider("Tier 1 threshold (Safe Raise)", 0.50, 0.90, 0.65, 0.01)
    t2 = st.slider("Tier 2 threshold (Test Raise)", 0.20, 0.80, 0.45, 0.01)
    if t2 >= t1:
        st.warning("Tier 2 threshold should be lower than Tier 1.")

    st.subheader("Action Filters")
    tier_filter = st.multiselect(
        "Show tiers",
        ["🟢 Tier 1 – Safe Raise", "🟡 Tier 2 – Test Raise", "🔴 Protect"],
        default=["🟢 Tier 1 – Safe Raise", "🟡 Tier 2 – Test Raise"]
    )
    min_uplift = st.number_input("Min Revenue Lift ($)", value=0, step=1000)

    st.subheader("Data Settings")
    n_rows = st.slider("Synthetic rows", 20000, 150000, 80000, 10000)
    seed = st.number_input("Random seed", value=42, step=1)

    show_debug = st.checkbox("Show debug panel", value=False)

# -----------------------------
# Data + Elasticity cube (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(n_rows: int, seed: int):
    df_raw = make_synthetic_transactions(n_rows=n_rows, seed=seed)
    cube = derive_elasticity_cube(df_raw)
    return df_raw, cube


with st.spinner("Building synthetic data + elasticity cube..."):
    df_raw, cube = load_data(n_rows, seed)

# -----------------------------
# Simulate price lift + score
# -----------------------------
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

# -----------------------------
# KPIs
# -----------------------------
total_uplift = float(sim_df["revenue_delta"].sum())
avg_el = float(sim_df["elasticity"].mean())
tier1_count = int((sim_df["raise_tier"] == "🟢 Tier 1 – Safe Raise").sum())
tier2_count = int((sim_df["raise_tier"] == "🟡 Tier 2 – Test Raise").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue Lift (All rows)", f"${total_uplift:,.0f}")
k2.metric("Average Elasticity", f"{avg_el:.2f}")
k3.metric("Tier 1 Safe Raises", f"{tier1_count:,}")
k4.metric("Tier 2 Test Raises", f"{tier2_count:,}")

st.divider()

# -----------------------------
# Visuals
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    fig_el = px.histogram(
        sim_df,
        x="elasticity",
        nbins=40,
        title="Elasticity Distribution (SKU × Segment × Region)"
    )
    st.plotly_chart(fig_el, use_container_width=True)

with c2:
    fig_score = px.histogram(
        sim_df,
        x="raise_score",
        nbins=40,
        title="Raise Score Distribution"
    )
    st.plotly_chart(fig_score, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    fig_scatter = px.scatter(
        sim_df,
        x="elasticity",
        y="avg_margin",
        color="raise_tier",
        hover_data=["sku", "segment", "region", "category", "raise_score", "revenue_delta", "vol_delta_pct"],
        title="Elasticity vs Margin (colored by Raise Tier)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with c4:
    fig_tiers = px.histogram(
        sim_df,
        x="raise_tier",
        title="Raise Tier Counts"
    )
    st.plotly_chart(fig_tiers, use_container_width=True)

st.divider()

# -----------------------------
# Action Table (ranked)
# -----------------------------
st.subheader("Recommended Actions (Ranked)")

action_df = sim_df[
    (sim_df["raise_tier"].isin(tier_filter)) &
    (sim_df["revenue_delta"] >= float(min_uplift))
].copy()

action_df = action_df.sort_values("revenue_delta", ascending=False)

show_cols = [
    "sku", "segment", "region", "category",
    "elasticity", "avg_margin",
    "avg_price", "new_price",
    "avg_units", "new_units",
    "vol_delta_pct",
    "base_revenue", "revenue_delta",
    "raise_score", "raise_tier"
]

st.dataframe(action_df[show_cols].head(50), use_container_width=True)

# -----------------------------
# Explain a Recommendation (drill-down)
# -----------------------------
st.subheader("Explain a Recommendation")

if len(action_df) == 0:
    st.info("No rows match your tier filter / minimum uplift. Adjust sidebar settings.")
else:
    # Add a key for selection (don’t mutate sim_df; keep it local)
    tmp = action_df.head(50).copy()
    tmp["key"] = tmp["sku"] + " | " + tmp["segment"] + " | " + tmp["region"]

    selected = st.selectbox("Select SKU / Segment / Region", tmp["key"].tolist())

    row = tmp[tmp["key"] == selected].iloc[0]

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Elasticity", f"{row['elasticity']:.2f}")
    e2.metric("Margin %", f"{row['avg_margin']*100:.1f}%")
    e3.metric("Volume Impact", f"{row['vol_delta_pct']:.1f}%")
    e4.metric("Revenue Lift", f"${row['revenue_delta']:,.0f}")

    st.write(
        f"**Why this recommendation?**  \n"
        f"- At **+{price_increase:.1f}%** price, expected units change is **{row['vol_delta_pct']:.1f}%**.  \n"
        f"- Demand sensitivity (elasticity) is **{row['elasticity']:.2f}** (less negative = safer).  \n"
        f"- Margin is **{row['avg_margin']*100:.1f}%** (healthier margin = better raise candidate).  \n"
        f"- Expected revenue uplift is **${row['revenue_delta']:,.0f}**.  \n"
        f"- Overall raise score = **{row['raise_score']:.2f}** → **{row['raise_tier']}**."
    )

st.divider()

# -----------------------------
# Optional Debug Panel
# -----------------------------
if show_debug:
    st.subheader("Debug Panel")

    d1, d2, d3 = st.columns(3)
    d1.metric("Raw rows", f"{len(df_raw):,}")
    d2.metric("Cube rows", f"{len(cube):,}")
    d3.metric("Sim rows", f"{len(sim_df):,}")

    st.write("Overall price-units correlation (raw):", float(df_raw["net_price"].corr(df_raw["units"])))

    st.write("Raw price std dev:", float(df_raw["net_price"].std()))
    st.write("Raw units std dev:", float(df_raw["units"].std()))

    # Quick sanity scatter
    sample = df_raw.sample(min(2000, len(df_raw)), random_state=1)
    fig_raw = px.scatter(sample, x="net_price", y="units", color="segment", title="Raw Price vs Units (sample)")
    st.plotly_chart(fig_raw, use_container_width=True)

st.caption("Note: This POC uses synthetic data for demonstration only.")
