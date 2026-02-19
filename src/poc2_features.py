import numpy as np
import pandas as pd

def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer-level features used for segmentation and leakage benchmarking.
    Expects df columns:
      customer_id, segment, region, category, list_price, net_price, unit_cost, units, contract_flag
    """
    d = df.copy()
    d["revenue"] = d["net_price"] * d["units"]
    d["gm"] = (d["net_price"] - d["unit_cost"]) * d["units"]
    d["discount_pct"] = (d["list_price"] - d["net_price"]) / (d["list_price"] + 1e-9)

    cust = d.groupby(["customer_id"]).agg(
        segment=("segment", "first"),
        region=("region", "first"),
        orders=("customer_id", "size"),
        sku_count=("sku", "nunique"),
        category_count=("category", "nunique"),
        total_units=("units", "sum"),
        total_revenue=("revenue", "sum"),
        total_gm=("gm", "sum"),
        avg_discount=("discount_pct", "mean"),
        p90_discount=("discount_pct", lambda x: float(np.quantile(x, 0.90))),
        contract_share=("contract_flag", "mean"),
    ).reset_index()

    cust["gm_pct"] = cust["total_gm"] / (cust["total_revenue"] + 1e-9)
    cust["aov"] = cust["total_revenue"] / (cust["orders"] + 1e-9)  # avg order value
    cust["units_per_order"] = cust["total_units"] / (cust["orders"] + 1e-9)
    cust["sku_per_order_proxy"] = cust["sku_count"] / (cust["orders"] + 1e-9)

    return cust
