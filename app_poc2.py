import streamlit as st
import plotly.express as px

from src.synth_data import make_synthetic_transactions
from src.poc2_features import build_customer_features
from src.poc2_segmentation import segment_customers, SEGMENT_FEATURES
from src.poc2_leakage import (
    leakage_flags,
    leakage_summary_by_customer,
    leakage_summary_by_rep,
)

st.set_page_config(page_title="Pricing Intelligence Engine – POC2", layout="wide")

st.title("AI-Driven Pricing Intelligence Engine")
st.caption("POC 2 — Customer Segmentation + Discount Leakage (Synthetic data)")

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    n_rows = st.slider("Synthetic rows", 20000, 150000, 80000, 10000)
    seed = st.number_input("Random seed", value=42, step=1)

    st.subheader("Segmentation")
    k = st.slider("Number of clusters (K)", 3, 10, 5, 1)

    st.subheader("Leakage Rules")
    percentile = st.slider("Leakage percentile threshold", 0.80, 0.99, 0.90, 0.01)
    min_peer_n = st.slider("Minimum peer transactions", 10, 200, 30, 10)
    st.caption("Leakage = discount above peer percentile (SKU×segment×region), when peer count ≥ minimum.")

# -----------------------------
# Load / Compute (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_all(n_rows, seed, k, percentile, min_peer_n):
    df = make_synthetic_transactions(n_rows=n_rows, seed=seed)

    # Customer features + segmentation
    cust = build_customer_features(df)
    cust_seg = segment_customers(cust, k=k)

    # Leakage
    txn_flagged = leakage_flags(df, percentile=percentile, min_peer_n=min_peer_n)
    cust_leak = leakage_summary_by_customer(txn_flagged)
    rep_leak = leakage_summary_by_rep(txn_flagged)

    return df, cust_seg, txn_flagged, cust_leak, rep_leak

df, cust_seg, txn_flagged, cust_leak, rep_leak = load_all(n_rows, seed, k, percentile, min_peer_n)

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{cust_seg.shape[0]:,}")
c2.metric("Leakage Txns", f"{int(txn_flagged['leakage_flag'].sum()):,}")
c3.metric("Est Leakage $", f"${cust_leak['leakage_est_dollars'].sum():,.0f}")
c4.metric("Avg Discount", f"{cust_seg['avg_discount'].mean()*100:.1f}%")

st.divider()

# -----------------------------
# Segmentation visuals
# -----------------------------
left, right = st.columns(2)

with left:
    fig1 = px.scatter(
        cust_seg,
        x="avg_discount",
        y="gm_pct",
        color="cluster_label",
        hover_data=["customer_id", "segment", "region", "total_revenue", "orders"],
        title="Customer Segments: Discount vs GM%"
    )
    st.plotly_chart(fig1, use_container_width=True)

with right:
    fig2 = px.bar(
        cust_seg.groupby("cluster_label", as_index=False)
               .agg(customers=("customer_id", "count"),
                    avg_disc=("avg_discount", "mean"),
                    avg_gm=("gm_pct", "mean"),
                    revenue=("total_revenue", "sum")),
        x="cluster_label",
        y="customers",
        title="Cluster Sizes"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Cluster Profile (Averages)")
profile = cust_seg.groupby("cluster_label")[SEGMENT_FEATURES].mean().reset_index()
st.dataframe(profile, use_container_width=True)

st.divider()

# -----------------------------
# Leakage by Customer
# -----------------------------
st.subheader("Leakage: Accounts to Review (Ranked)")

show_cols = [
    "customer_id", "segment", "region",
    "leakage_txns", "leakage_est_dollars",
    "avg_discount", "p90_discount", "gm_pct", "revenue"
]
st.dataframe(cust_leak[show_cols].head(50), use_container_width=True)

st.subheader("Explain Leakage (pick a customer)")
cust_list = cust_leak["customer_id"].head(50).tolist()

if len(cust_list) == 0:
    st.info("No customers flagged under current leakage rules. Try lowering the percentile or min peer threshold.")
else:
    selected = st.selectbox("Select customer", cust_list)

    tx = txn_flagged[txn_flagged["customer_id"] == selected].copy()
    tx_leak = tx[tx["leakage_flag"]].sort_values("leakage_dollars_est", ascending=False).head(25)

    e1, e2, e3 = st.columns(3)
    e1.metric("Leakage Txns", f"{int(tx['leakage_flag'].sum()):,}")
    e2.metric("Est Leakage $", f"${tx['leakage_dollars_est'].sum():,.0f}")
    e3.metric("Avg Discount", f"{tx['discount_pct'].mean()*100:.1f}%")

    st.write("Top leakage transactions (by estimated $ impact):")
    st.dataframe(
        tx_leak[[
            "sales_rep_id",
            "sku", "category", "segment", "region",
            "list_price", "net_price", "units",
            "discount_pct", "peer_q_disc", "peer_n",
            "leakage_dollars_est"
        ]].head(25),
        use_container_width=True
    )

st.divider()

# -----------------------------
# Leakage by Rep (NEW)
# -----------------------------
st.subheader("Leakage: Rep Leaderboard (Ranked)")

rep_cols = [
    "sales_rep_id", "customers", "skus",
    "leakage_txns", "leakage_est_dollars",
    "avg_discount", "gm_pct", "revenue"
]
st.dataframe(rep_leak[rep_cols].head(30), use_container_width=True)

st.subheader("Explain Rep Leakage (pick a rep)")
rep_list = rep_leak["sales_rep_id"].head(30).tolist()

if len(rep_list) == 0:
    st.info("No reps flagged under current leakage rules. Try lowering the percentile or min peer threshold.")
else:
    selected_rep = st.selectbox("Select rep", rep_list)

    rep_tx = txn_flagged[txn_flagged["sales_rep_id"] == selected_rep].copy()
    rep_tx_leak = rep_tx[rep_tx["leakage_flag"]].sort_values("leakage_dollars_est", ascending=False).head(25)

    r1, r2, r3 = st.columns(3)
    r1.metric("Leakage Txns", f"{int(rep_tx['leakage_flag'].sum()):,}")
    r2.metric("Est Leakage $", f"${rep_tx['leakage_dollars_est'].sum():,.0f}")
    r3.metric("Avg Discount", f"{rep_tx['discount_pct'].mean()*100:.1f}%")

    st.write("Top leakage transactions for this rep:")
    st.dataframe(
        rep_tx_leak[[
            "customer_id",
            "sku", "category", "region",
            "list_price", "net_price", "units",
            "discount_pct", "peer_q_disc", "peer_n",
            "leakage_dollars_est"
        ]].head(25),
        use_container_width=True
    )

st.caption("Note: This POC uses synthetic data for demonstration only.")
