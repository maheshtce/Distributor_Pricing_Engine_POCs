import numpy as np
import pandas as pd

def make_synthetic_transactions(n_rows=50000, seed=42):
    np.random.seed(seed)

    skus = [f"SKU_{i}" for i in range(1, 301)]
    segments = ["DSO", "Clinic", "Small Practice", "Hospital"]
    regions = ["Northeast", "South", "Midwest", "West"]
    categories = ["Dental", "MedSurg", "Lab"]

    data = {
        "sku": np.random.choice(skus, n_rows),
        "segment": np.random.choice(segments, n_rows),
        "region": np.random.choice(regions, n_rows),
        "category": np.random.choice(categories, n_rows),
        "net_price": np.random.uniform(5, 100, n_rows),
        "unit_cost": np.random.uniform(3, 70, n_rows),
        "units": np.random.poisson(50, n_rows),
        "contract_flag": np.random.choice([0,1], n_rows, p=[0.6,0.4])
    }

    df = pd.DataFrame(data)
    df["margin_pct"] = (df["net_price"] - df["unit_cost"]) / df["net_price"]
    return df

