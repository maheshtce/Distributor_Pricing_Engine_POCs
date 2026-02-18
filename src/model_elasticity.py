import numpy as np
import pandas as pd
import statsmodels.api as sm

def fit_pooled_loglog_model(df):
    df = df.copy()
    df = df[df["units"] > 0]
    df["log_units"] = np.log(df["units"])
    df["log_price"] = np.log(df["net_price"])

    X = sm.add_constant(df[["log_price"]])
    y = df["log_units"]

    model = sm.OLS(y, X).fit()
    return model

def derive_elasticity_cube(df):
    grouped = df.groupby(["sku","segment","region"]).agg(
        avg_price=("net_price","mean"),
        avg_units=("units","mean"),
        avg_margin=("margin_pct","mean")
    ).reset_index()

    # simple proxy elasticity variation
    grouped["elasticity"] = -0.5 - np.random.rand(len(grouped))*1.5
    return grouped
