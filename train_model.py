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
LOW_Q = 0.01
HIGH_Q = 0.99


def qrange(series: pd.Series, low_q=LOW_Q, high_q=HIGH_Q) -> tuple[float, float]:
    """Robust min/max using quantiles (better for UI sliders)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (0.0, 1.0)

    lo = float(s.quantile(low_q))
    hi = float(s.quantile(high_q))

    if lo == hi:
        lo = float(s.min())
        hi = float(s.max())

    if lo > hi:
        lo, hi = hi, lo

    return (lo, hi)


def main():
    df = pd.read_csv(DATA_PATH)

    needed = ["open", "low", "close", "volume", "commodity", "high"]
    df = df.dropna(subset=needed).copy()

    # Convert numeric columns
    for c in ["open", "low", "close", "high", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "low", "close", "high", "volume"]).copy()

    # Remove negative price rows (treated as anomalies for this project)
    price_cols = ["open", "low", "close", "high"]
    df = df[(df[price_cols] >= 0).all(axis=1)].copy()

    # Ensure volume is positive (for log transform)
    df = df[df["volume"] > 0].copy()

    # Keep commodity as string for UI and grouping
    df["commodity_str"] = df["commodity"].astype(str)

    # Encode commodity
    le = LabelEncoder()
    df["commodity"] = le.fit_transform(df["commodity_str"])
    encoders = {"commodity": le}

    # Log-transform volume to reduce extreme scale
    df["volume_log10"] = np.log10(df["volume"].clip(lower=1))

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

    # Store feature importances directly in the bundle
    feature_importances = model.feature_importances_.tolist()

    # UI ranges per commodity (robust quantiles)
    ui_ranges_by_commodity: dict[str, dict[str, tuple[float, float]]] = {}
    for comm_name, g in df.groupby("commodity_str"):
        ui_ranges_by_commodity[comm_name] = {
            "open": qrange(g["open"]),
            "low": qrange(g["low"]),
            "close": qrange(g["close"]),
            "volume_log10": qrange(g["volume_log10"]),
        }

    # Global ranges (fallback)
    ui_ranges_global = {
        "open": qrange(df["open"]),
        "low": qrange(df["low"]),
        "close": qrange(df["close"]),
        "volume_log10": qrange(df["volume_log10"]),
    }

    # Legacy key kept for backward compatibility
    ui_feature_ranges = dict(ui_ranges_global)

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
        "feature_importances": feature_importances,
        "ui_ranges_by_commodity": ui_ranges_by_commodity,
        "ui_ranges_global": ui_ranges_global,
        "ui_feature_ranges": ui_feature_ranges,  # legacy
        "raw_feature_ranges": raw_feature_ranges,
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
            "rows_after_cleaning": int(len(df)),
            "note": "Negative price rows removed as anomalies for this project.",
        },
    }

    joblib.dump(bundle, OUTPUT_PATH)

    print(f"Model saved to {OUTPUT_PATH}")
    print("Bundle keys:", sorted(bundle.keys()))
    print("Commodities:", len(ui_ranges_by_commodity))


if __name__ == "__main__":
    main()
