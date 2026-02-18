import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def _loglog_elasticity(g: pd.DataFrame, min_rows: int, min_unique_prices: int) -> float | None:
    g = g[(g["units"] > 0) & (g["net_price"] > 0)].copy()
    if len(g) < min_rows:
        return None
    if g["net_price"].nunique() < min_unique_prices:
        return None

    X = np.log(g[["net_price"]].values)
    y = np.log(g["units"].values)

    # Fit log-log demand curve
    lr = LinearRegression()
    lr.fit(X, y)
    return float(lr.coef_[0])

def derive_elasticity_cube(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns cube with elasticity at SKU×segment×region using raw transactions,
    with fallbacks for sparse groups.
    """
    df = df.copy()
    df["margin_pct"] = (df["net_price"] - df["unit_cost"]) / df["net_price"]

    # Summary stats at the target grain
    keys = ["sku", "segment", "region"]
    summary = df.groupby(keys).agg(
        avg_price=("net_price", "mean"),
        avg_units=("units", "mean"),
        avg_margin=("margin_pct", "mean"),
        category=("category", "first"),
        n=("units", "size"),
        unique_prices=("net_price", "nunique"),
    ).reset_index()

    # 1) SKU×segment×region elasticity
    esr = []
    for k, g in df.groupby(keys):
        e = _loglog_elasticity(g, min_rows=60, min_unique_prices=6)
        esr.append((*k, e))
    esr = pd.DataFrame(esr, columns=["sku", "segment", "region", "e_sku_seg_reg"])

    # 2) SKU×segment fallback
    ess = []
    for (sku, seg), g in df.groupby(["sku", "segment"]):
        e = _loglog_elasticity(g, min_rows=150, min_unique_prices=8)
        ess.append((sku, seg, e))
    ess = pd.DataFrame(ess, columns=["sku", "segment", "e_sku_seg"])

    # 3) SKU-only fallback
    esk = []
    for sku, g in df.groupby(["sku"]):
        e = _loglog_elasticity(g, min_rows=300, min_unique_prices=10)
        esk.append((sku, e))
    esk = pd.DataFrame(esk, columns=["sku", "e_sku"])

    # 4) segment×region fallback
    esr2 = []
    for (seg, reg), g in df.groupby(["segment", "region"]):
        e = _loglog_elasticity(g, min_rows=800, min_unique_prices=10)
        esr2.append((seg, reg, e))
    esr2 = pd.DataFrame(esr2, columns=["segment", "region", "e_seg_reg"])

    # 5) global fallback
    e_global = _loglog_elasticity(df, min_rows=3000, min_unique_prices=15)
    if e_global is None:
        e_global = -1.0  # sane default

    out = (
        summary
        .merge(esr, on=keys, how="left")
        .merge(ess, on=["sku", "segment"], how="left")
        .merge(esk, on=["sku"], how="left")
        .merge(esr2, on=["segment", "region"], how="left")
    )

    # Final elasticity with fallbacks
    out["elasticity"] = out["e_sku_seg_reg"]
    out["elasticity"] = out["elasticity"].fillna(out["e_sku_seg"])
    out["elasticity"] = out["elasticity"].fillna(out["e_sku"])
    out["elasticity"] = out["elasticity"].fillna(out["e_seg_reg"])
    out["elasticity"] = out["elasticity"].fillna(e_global)

    # Clip to plausible band for distribution products
    out["elasticity"] = out["elasticity"].clip(-4.0, -0.05)

    return out
