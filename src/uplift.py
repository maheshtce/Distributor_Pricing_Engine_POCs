import numpy as np

def compute_price_lift_impact(df, price_increase_pct):
    df = df.copy()

    df["new_price"] = df["avg_price"] * (1 + price_increase_pct/100)

    # ΔQ = elasticity × %ΔP
    df["new_units"] = df["avg_units"] * (
        1 + df["elasticity"] * (price_increase_pct/100)
    )

    df["base_revenue"] = df["avg_price"] * df["avg_units"]
    df["new_revenue"] = df["new_price"] * df["new_units"]

    df["revenue_delta"] = df["new_revenue"] - df["base_revenue"]

   # Normalize elasticity (assume range -4 to 0)
    df["elasticity_norm"] = (df["elasticity"] + 4) / 4
    df["elasticity_norm"] = df["elasticity_norm"].clip(0, 1)
    
    # Margin already between 0 and 1
    df["margin_norm"] = df["avg_margin"].clip(0, 1)
    
    # Revenue uplift % normalized
    df["rev_uplift_pct"] = df["revenue_delta"] / df["base_revenue"]
    df["rev_uplift_norm"] = (df["rev_uplift_pct"] - df["rev_uplift_pct"].min()) / (
        df["rev_uplift_pct"].max() - df["rev_uplift_pct"].min() + 1e-6
    )
    
    # Volume risk penalty (bigger drop = worse)
    df["vol_risk_norm"] = np.clip(-df["vol_delta_pct"] / 10, 0, 1)
    
    # Weighted Raise Score
    df["raise_score"] = (
        0.35 * df["elasticity_norm"] +
        0.30 * df["margin_norm"] +
        0.25 * df["rev_uplift_norm"] -
        0.10 * df["vol_risk_norm"]
    )

def assign_tier(score):
    if score >= 0.65:
        return "🟢 Tier 1 – Safe Raise"
    elif score >= 0.45:
        return "🟡 Tier 2 – Test Raise"
    else:
        return "🔴 Protect"

df["raise_tier"] = df["raise_score"].apply(assign_tier)
    

    return df
