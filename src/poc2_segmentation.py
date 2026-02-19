import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEGMENT_FEATURES = [
    "orders", "sku_count", "category_count",
    "total_revenue", "gm_pct",
    "avg_discount", "p90_discount",
    "contract_share", "aov", "units_per_order"
]

def segment_customers(cust_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    d = cust_df.copy()

    X = d[SEGMENT_FEATURES].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    d["cluster"] = km.fit_predict(Xs)

    # add readable labels (simple)
    d["cluster_label"] = d["cluster"].apply(lambda c: f"Cluster {c}")

    return d
