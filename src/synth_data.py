import numpy as np
import pandas as pd

def make_synthetic_transactions(n_rows=80000, seed=42):
    rng = np.random.default_rng(seed)

    skus = [f"SKU_{i}" for i in range(1, 301)]
    segments = ["DSO", "Clinic", "Small Practice", "Hospital"]
    regions = ["Northeast", "South", "Midwest", "West"]
    categories = ["Dental", "MedSurg", "Lab"]

    # Segment-level elasticity priors (DSOs tend to be more price sensitive)
    seg_el = {"DSO": -1.6, "Clinic": -1.2, "Small Practice": -1.0, "Hospital": -0.8}
    # Region tweaks
    reg_adj = {"Northeast": -0.1, "South": -0.2, "Midwest": 0.0, "West": -0.15}

    sku = rng.choice(skus, n_rows)
    segment = rng.choice(segments, n_rows)
    region = rng.choice(regions, n_rows)
    category = rng.choice(categories, n_rows)

    # Base price per SKU (lognormal gives a realistic skew)
    sku_base_price = {s: float(rng.lognormal(mean=3.1, sigma=0.5)) for s in skus}  # ~ $10-$80 typical
    base_price = np.array([sku_base_price[s] for s in sku])

    # Price varies around base price (promos / negotiations)
    promo_shock = rng.normal(0, 0.12, n_rows)  # ~ +/- 12% typical
    net_price = np.clip(base_price * np.exp(promo_shock), 2.0, None)

    # Unit cost correlated with price
    unit_cost = np.clip(net_price * rng.uniform(0.55, 0.80, n_rows), 1.0, None)

    contract_flag = rng.choice([0, 1], n_rows, p=[0.6, 0.4])

    # True elasticity per row (segment + region + sku noise)
    true_el = np.array([seg_el[s] for s in segment]) + np.array([reg_adj[r] for r in region])
    true_el += rng.normal(0, 0.15, n_rows)  # sku/account randomness

    # Base demand scale depends on category + segment
    cat_scale = np.where(category == "Dental", 55, np.where(category == "MedSurg", 70, 45))
    seg_scale = np.where(segment == "DSO", 110, np.where(segment == "Hospital", 85, 60))

    # Demand model: units = scale * (price/base_price)^elasticity * noise
    noise = rng.lognormal(mean=0, sigma=0.35, size=n_rows)
    expected_units = (cat_scale * seg_scale/80) * (net_price / base_price) ** (true_el) * noise

    # Convert to integers (poisson around expected)
    units = rng.poisson(lam=np.clip(expected_units, 0.2, 500))

    df = pd.DataFrame({
        "sku": sku,
        "segment": segment,
        "region": region,
        "category": category,
        "net_price": net_price,
        "unit_cost": unit_cost,
        "units": units,
        "contract_flag": contract_flag,
    })
    df["margin_pct"] = (df["net_price"] - df["unit_cost"]) / df["net_price"]
    return df
