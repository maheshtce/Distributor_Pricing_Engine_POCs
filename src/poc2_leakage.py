import numpy as np
import pandas as pd

def leakage_flags(df: pd.DataFrame, percentile: float = 0.90, min_peer_n: int = 30) -> pd.DataFrame:
    """
    Transaction-level leakage flags using peer benchmark:
      Peer group = SKU × segment × region
      Leakage if discount_pct > peer_percentile (and peer_n >= min_peer_n)

    Requires columns:
      sku, customer_id, sales_rep_id, segment, region, list_price, net_price, unit_cost, units
    """
    d = df.copy()
    d["discount_pct"] = (d["list_price"] - d["net_price"]) / (d["list_price"] + 1e-9)
    d["gm_pct_txn"] = (d["net_price"] - d["unit_cost"]) / (d["net_price"] + 1e-9)
    d["revenue"] = d["net_price"] * d["units"]
    d["gm"] = (d["net_price"] - d["unit_cost"]) * d["units"]

    def q_func(x):
        return float(np.quantile(x, percentile))

    peer = d.groupby(["sku", "segment", "region"]).agg(
        peer_avg_disc=("discount_pct", "mean"),
        peer_q_disc=("discount_pct", q_func),
        peer_avg_gm=("gm_pct_txn", "mean"),
        peer_n=("discount_pct", "size"),
    ).reset_index()

    out = d.merge(peer, on=["sku", "segment", "region"], how="left")

    out["leakage_flag"] = (out["peer_n"] >= min_peer_n) & (out["discount_pct"] > out["peer_q_disc"])

    # Excess discount % over peer threshold
    out["excess_disc_pct"] = (out["discount_pct"] - out["peer_q_disc"]).clip(lower=0)

    # Estimated $ impact (simple proxy): excess % * list * units
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


def leakage_summary_by_rep(txn_flagged: pd.DataFrame) -> pd.DataFrame:
    d = txn_flagged.copy()
    rep = d.groupby("sales_rep_id").agg(
        leakage_txns=("leakage_flag", "sum"),
        leakage_est_dollars=("leakage_dollars_est", "sum"),
        avg_discount=("discount_pct", "mean"),
        revenue=("revenue", "sum"),
        gm=("gm", "sum"),
        customers=("customer_id", "nunique"),
        skus=("sku", "nunique"),
    ).reset_index()
    rep["gm_pct"] = rep["gm"] / (rep["revenue"] + 1e-9)
    rep["leakage_rate"] = rep["leakage_txns"] / (len(d) + 1e-9)  # simple, mostly for display
    return rep.sort_values("leakage_est_dollars", ascending=False)
