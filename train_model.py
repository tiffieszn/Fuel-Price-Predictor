# train_model.py
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "all_fuels_data.csv"
OUTPUT_PATH = "model.pkl"

RANDOM_STATE = 42

# Robust bounds for UI sliders (avoids extreme outliers dominating the slider)
LOW_Q = 0.01
HIGH_Q = 0.99


def qrange(series: pd.Series, low_q=LOW_Q, high_q=HIGH_Q) -> tuple[float, float]:
    """Robust min/max for UI sliders using quantiles."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (0.0, 1.0)
    lo = float(s.quantile(low_q))
    hi = float(s.quantile(high_q))
    if lo == hi:
        lo = float(s.min())
        hi = float(s.max())
    return (lo, hi)


def main():
    df = pd.read_csv(DATA_PATH)

    # Keep only needed columns, drop missing
    needed = ["open", "low", "close", "volume", "commodity", "high"]
    df = df.dropna(subset=needed).copy()

    # Coerce numeric
    for c in ["open", "low", "close", "high", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "low", "close", "high", "volume"]).copy()

    # ====== Remove negative prices (per your request) ======
    # You said negative values are likely a 1-row anomaly.
    # We remove ANY rows where price columns are negative.
    price_cols = ["open", "low", "close", "high"]
    df = df[(df[price_cols] >= 0).all(axis=1)].copy()

    # Also remove non-positive volumes (can't log)
    df = df[df["volume"] > 0].copy()

    # Encode commodity (keep original string for UI ranges)
    df["commodity_str"] = df["commodity"].astype(str)

    le = LabelEncoder()
    df["commodity"] = le.fit_transform(df["commodity_str"])
    encoders = {"commodity": le}

    # Volume transform for training stability
    df["volume_log10"] = np.log10(df["volume"].clip(lower=1))

    # Define model features/target
    features = ["open", "low", "close", "volume_log10", "commodity"]
    target = "high"

    X = df[features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = GradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        n_estimators=500,
        max_depth=5,
        random_state=RANDOM_STATE,
        max_features=None,
    )
    model.fit(x_train, y_train)

    # ====== Per-commodity UI ranges (robust quantiles) ======
    # We store ranges by COMMODITY NAME (string), because that's what user selects in the UI.
    ui_ranges_by_commodity: dict[str, dict[str, tuple[float, float]]] = {}

    for comm_name, g in df.groupby("commodity_str"):
        ui_ranges_by_commodity[comm_name] = {
            "open": qrange(g["open"]),
            "low": qrange(g["low"]),
            "close": qrange(g["close"]),
            # slider will be in LOG space (much more user-friendly)
            "volume_log10": qrange(g["volume_log10"]),
        }

    # Fallback/global ranges (in case something goes weird)
    ui_ranges_global = {
        "open": qrange(df["open"]),
        "low": qrange(df["low"]),
        "close": qrange(df["close"]),
        "volume_log10": qrange(df["volume_log10"]),
    }

    # Raw ranges (reference/debug)
    raw_feature_ranges = {
        "open": (float(df["open"].min()), float(df["open"].max())),
        "low": (float(df["low"].min()), float(df["low"].max())),
        "close": (float(df["close"].min()), float(df["close"].max())),
        "high": (float(df["high"].min()), float(df["high"].max())),
        "volume": (float(df["volume"].min()), float(df["volume"].max())),
    }

    bundle = {
        "model": model,
        "encoders": encoders,
        "features": features,
        "target": target,
        # UI ranges for Streamlit sliders
        "ui_ranges_by_commodity": ui_ranges_by_commodity,
        "ui_ranges_global": ui_ranges_global,
        # Raw ranges
        "raw_feature_ranges": raw_feature_ranges,
        # Tell the app how to transform volume
        "transforms": {
            "volume": {
                "type": "log10",
                "clip_lower": 1,
                "feature_name": "volume_log10",
                "inverse": "10**x",
            }
        },
        "meta": {
            "data_path": DATA_PATH,
            "quantiles_for_ui": (LOW_Q, HIGH_Q),
            "random_state": RANDOM_STATE,
            "note": "Negative prices removed for UI/domain constraints.",
        },
    }

    joblib.dump(bundle, OUTPUT_PATH)

    print(f"Model trained and saved as {OUTPUT_PATH}")
    print(f"Rows after cleaning: {len(df):,}")
    print("Example UI ranges (global):")
    for k, v in ui_ranges_global.items():
        print(f"  - {k}: {v}")
    print(f"Commodities saved: {len(ui_ranges_by_commodity)}")


if __name__ == "__main__":
    main()
