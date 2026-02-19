import numpy as np

def assign_tier(score: float, t1: float, t2: float) -> str:
    if score >= t1:
        return "🟢 Tier 1 – Safe Raise"
    elif score >= t2:
        return "🟡 Tier 2 – Test Raise"
    else:
        return "🔴 Protect"

def compute_price_lift_impact(
    df,
    price_increase_pct: float,
    w_elasticity: float = 0.35,
    w_margin: float = 0.30,
    w_rev_uplift: float = 0.25,
    w_vol_risk: float = 0.10,   # penalty weight
    t1: float = 0.65,
    t2: float = 0.45,
):
    df = df.copy()

    # New price
    df["new_price"] = df["avg_price"] * (1 + price_increase_pct / 100)

    # New units using elasticity (ΔQ% = elasticity * ΔP%)
    df["new_units"] = df["avg_units"] * (1 + df["elasticity"] * (price_increase_pct / 100))

    # Revenue
    df["base_revenue"] = df["avg_price"] * df["avg_units"]
    df["new_revenue"] = df["new_price"] * df["new_units"]
    df["revenue_delta"] = df["new_revenue"] - df["base_revenue"]

    # Volume change %
    df["vol_delta_pct"] = (df["new_units"] - df["avg_units"]) / (df["avg_units"] + 1e-9) * 100

    # ---------- Normalize score components ----------
    # Elasticity: assume plausible band [-4, 0] (less negative is better)
    df["elasticity_norm"] = ((df["elasticity"] + 4) / 4).clip(0, 1)

    # Margin: already [0,1]
    df["margin_norm"] = df["avg_margin"].clip(0, 1)

    # Revenue uplift % normalized to [0,1] across rows
    df["rev_uplift_pct"] = df["revenue_delta"] / (df["base_revenue"] + 1e-9)
    rev_min = df["rev_uplift_pct"].min()
    rev_max = df["rev_uplift_pct"].max()
    df["rev_uplift_norm"] = (df["rev_uplift_pct"] - rev_min) / (rev_max - rev_min + 1e-6)

    # Volume risk penalty: larger drop => higher penalty (0..1)
    df["vol_risk_norm"] = np.clip(-df["vol_delta_pct"] / 10.0, 0, 1)

    # ---------- Raise score ----------
    df["raise_score"] = (
        w_elasticity * df["elasticity_norm"]
        + w_margin * df["margin_norm"]
        + w_rev_uplift * df["rev_uplift_norm"]
        - w_vol_risk * df["vol_risk_norm"]
    ).clip(0, 1)

    # Tier assignment
    df["raise_tier"] = df["raise_score"].apply(lambda s: assign_tier(s, t1, t2))

    return df
