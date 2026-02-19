import streamlit as st
import plotly.express as px

from src.synth_data import make_synthetic_transactions
from src.poc2_features import build_customer_features
from src.poc2_segmentation import segment_customers, SEGMENT_FEATURES
from src.poc2_leakage import leakage_flags, leakage_summary_by_customer

st.set_page_config(page_title="Pricing Intelligence Engine – POC2", layout="wide")

st.title("AI-Driven Pricing Intelligence Engine")
st.caption("POC 2 — Customer Segmentation + Discount Leakage (Synthetic data)")

with st.sidebar:
    st.header("Controls")
    n_rows = st.slider("Synthetic rows", 20000, 150000, 80000, 10000)
    seed = st.number_input("Random seed", value=42, step=1)

    st.subheader("Segmentation")
    k = st.slider("Number of clusters (K)", 3, 10, 5, 1)

    st.subheader("Leakage Rules")
    st.write("Leakage = discount above peer 90th percentile (SKU×segment×region, n>=30).")

@st.cache_data(show_spinner=False)
def load_all(n_rows, seed, k):
    df = make_synthetic_transactions(n_rows=n_rows, seed=seed)

    # Build customer feature table
    cust = build_customer_features(df)
    cust_seg = segment_customers(cust, k=k)

    # Leakage
    txn_flagged = leakage_flags(df)
    cust_leak = leakage_summary_by_customer(txn_flagged)

    return df, cust_seg, txn_flagged, cust_leak

df, cust_seg, txn_flagged, cust_leak = load_all(n_rows, seed, k)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{cust_seg.shape[0]:,}")
c2.metric("Leakage Txns", f"{int(txn_flagged['leakage_flag'].sum()):,}")
c3.metric("Est Leakage $", f"${cust_leak['leakage_est_dollars'].sum():,.0f}")
c4.metric("Avg Discount", f"{cust_seg['avg_discount'].mean()*100:.1f}%")

st.divider()

# Segmentation visuals
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
               .agg(customers=("customer_id","count"),
                    avg_disc=("avg_discount","mean"),
                    avg_gm=("gm_pct","mean"),
                    revenue=("total_revenue","sum")),
        x="cluster_label",
        y="customers",
        title="Cluster Sizes"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Cluster Profile (Averages)")
profile = cust_seg.groupby("cluster_label")[SEGMENT_FEATURES].mean().reset_index()
st.dataframe(profile, use_container_width=True)

st.divider()

# Leakage section
st.subheader("Leakage: Accounts to Review (Ranked)")

show_cols = [
    "customer_id","segment","region",
    "leakage_txns","leakage_est_dollars",
    "avg_discount","p90_discount","gm_pct","revenue"
]
st.dataframe(cust_leak[show_cols].head(50), use_container_width=True)

st.subheader("Explain Leakage (pick a customer)")
cust_list = cust_leak["customer_id"].head(50).tolist()
selected = st.selectbox("Select customer", cust_list)

tx = txn_flagged[txn_flagged["customer_id"] == selected].copy()
tx_leak = tx[tx["leakage_flag"]].sort_values("leakage_dollars_est", ascending=False).head(25)

e1, e2, e3 = st.columns(3)
e1.metric("Leakage Txns", f"{int(tx['leakage_flag'].sum()):,}")
e2.metric("Est Leakage $", f"${tx['leakage_dollars_est'].sum():,.0f}")
e3.metric("Avg Discount", f"{tx['discount_pct'].mean()*100:.1f}%")

st.write("Top leakage transactions (by estimated $ impact):")
st.dataframe(
    tx_leak[["sku","category","segment","region","list_price","net_price","units","discount_pct","peer_p90_disc","leakage_dollars_est"]]
    .head(25),
    use_container_width=True
)

st.caption("Note: This POC uses synthetic data for demonstration only.")
