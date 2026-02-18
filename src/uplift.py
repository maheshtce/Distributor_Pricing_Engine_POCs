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

    return df
