import numpy as np
import pandas as pd

def leakage_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction-level leakage flags.
    Expects: sku, customer_id, segment, region, list_price, net_price, unit_cost, units
    """
    d = df.copy()
    d["discount_pct"] = (d["list_price"] - d["net_price"]) / (d["list_price"] + 1e-9)
    d["gm_pct_txn"] = (d["net_price"] - d["unit_cost"]) / (d["net_price"] + 1e-9)
    d["revenue"] = d["net_price"] * d["units"]
    d["gm"] = (d["net_price"] - d["unit_cost"]) * d["units"]

    # Peer benchmark: SKU × segment × region typical discount
    peer = d.groupby(["sku", "segment", "region"]).agg(
        peer_avg_disc=("discount_pct", "mean"),
        peer_p90_disc=("discount_pct", lambda x: float(np.quantile(x, 0.90))),
        peer_avg_gm=("gm_pct_txn", "mean"),
        n=("discount_pct", "size")
    ).reset_index()

    out = d.merge(peer, on=["sku", "segment", "region"], how="left")

    # Leakage definition: discount above peer 90th percentile (and enough peers)
    out["leakage_flag"] = (out["n"] >= 30) & (out["discount_pct"] > out["peer_p90_disc"])

    # $ impact = "excess discount" * list_price * units (simple proxy)
    out["excess_disc_pct"] = (out["discount_pct"] - out["peer_p90_disc"]).clip(lower=0)
    out["leakage_dollars_est"] = out["excess_disc_pct"] * out["list_price"] * out["units"]

    return out

def leakage_summary_by_customer(txn_flagged: pd.DataFrame) -> pd.DataFrame:
    d = txn_flagged.copy()
    cust = d.groupby("customer_id").agg(
        segment=("segment", "first"),
        region=("region", "first"),
        leakage_txns=("leakage_flag", "sum"),
        leakage_est_dollars=("leakage_dollars_est", "sum"),
        avg_discount=("discount_pct", "mean"),
        p90_discount=("discount_pct", lambda x: float(np.quantile(x, 0.90))),
        revenue=("revenue", "sum"),
        gm=("gm", "sum"),
    ).reset_index()
    cust["gm_pct"] = cust["gm"] / (cust["revenue"] + 1e-9)
    return cust.sort_values("leakage_est_dollars", ascending=False)
