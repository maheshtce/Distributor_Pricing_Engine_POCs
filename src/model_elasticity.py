import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def _fit_elasticity_loglog(group: pd.DataFrame, min_rows: int = 40) -> float | None:
    """
    Fits: log(units) = a + b*log(price)
    Returns b (elasticity). If insufficient data/variation, returns None.
    """
    g = group.copy()
    g = g[(g["units"] > 0) & (g["net_price"] > 0)]

    if len(g) < min_rows:
        return None

    # Need price variation, otherwise regression is meaningless
    if g["net_price"].nunique() < 5:
        return None

    X = np.log(g[["net_price"]].values)
    y = np.log(g["units"].values)

    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0])

def derive_elasticity_cube(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces SKU × segment × region cube with elasticity + summary stats.
    """
    df = df.copy()
    df["margin_pct"] = (df["net_price"] - df["unit_cost"]) / df["net_price"]

    # 1) Fit group elasticities
    keys = ["sku", "segment", "region"]
    elasticities = []
    for (sku, seg, reg), g in df.groupby(keys):
        e = _fit_elasticity_loglog(g)
        elasticities.append((sku, seg, reg, e, len(g)))

    e_df = pd.DataFrame(elasticities, columns=["sku", "segment", "region", "elasticity_raw", "n_rows"])

    # 2) Fallbacks: if group elasticity missing, use segment×region, else global
    # segment×region fallback
    sr_el = []
    for (seg, reg), g in df.groupby(["segment", "region"]):
        e = _fit_elasticity_loglog(g, min_rows=200)
        sr_el.append((seg, reg, e))
    sr_df = pd.DataFrame(sr_el, columns=["segment", "region", "elasticity_sr"])

    # global fallback
    global_e = _fit_elasticity_loglog(df, min_rows=500) or -1.0

    # 3) Merge summary stats
    summary = df.groupby(keys).agg(
        avg_price=("net_price", "mean"),
        avg_units=("units", "mean"),
        avg_margin=("margin_pct", "mean"),
        category=("category", "first"),
    ).reset_index()

    out = summary.merge(e_df, on=keys, how="left").merge(sr_df, on=["segment", "region"], how="left")

    # final elasticity
    out["elasticity"] = out["elasticity_raw"]
    out["elasticity"] = out["elasticity"].fillna(out["elasticity_sr"])
    out["elasticity"] = out["elasticity"].fillna(global_e)

    # optional clipping to keep sane bounds for dashboards
    out["elasticity"] = out["elasticity"].clip(lower=-6.0, upper=1.0)

    return out
