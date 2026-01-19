# train_model.py
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_PATH = "all_fuels_data.csv"
OUTPUT_PATH = "model.pkl"

LOW_Q = 0.01
HIGH_Q = 0.99

RANDOM_STATE = 42


def qrange(series: pd.Series, low_q=LOW_Q, high_q=HIGH_Q) -> tuple[float, float]:
    """Robust min/max for UI sliders using quantiles."""
    lo = float(series.quantile(low_q))
    hi = float(series.quantile(high_q))
    if lo == hi:
        lo = float(series.min())
        hi = float(series.max())
    return (lo, hi)


def main():
    df = pd.read_csv(DATA_PATH)

    needed = ["open", "low", "close", "volume", "commodity", "high"]
    df = df.dropna(subset=needed).copy()

    encoders = {}
    le = LabelEncoder()
    df["commodity"] = le.fit_transform(df["commodity"].astype(str))
    encoders["commodity"] = le

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["volume"]).copy()
    df["volume_clipped"] = df["volume"].clip(lower=1)
    df["volume_log10"] = np.log10(df["volume_clipped"])

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

    ui_feature_ranges = {
        "open": qrange(df["open"]),
        "low": qrange(df["low"]),
        "close": qrange(df["close"]),
        "volume_log10": qrange(df["volume_log10"]),
    }

    raw_feature_ranges = {
        "open": (float(df["open"].min()), float(df["open"].max())),
        "low": (float(df["low"].min()), float(df["low"].max())),
        "close": (float(df["close"].min()), float(df["close"].max())),
        "volume": (float(df["volume"].min()), float(df["volume"].max())),
    }

    bundle = {
        "model": model,
        "encoders": encoders,
        "features": features,
        "target": target,
        "ui_feature_ranges": ui_feature_ranges,
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
        },
    }

    joblib.dump(bundle, OUTPUT_PATH)

    print(f"Model trained and saved as {OUTPUT_PATH}")
    print("Saved UI feature ranges (robust quantiles):")
    for k, v in ui_feature_ranges.items():
        print(f"  - {k}: {v}")
    print("Saved raw feature ranges (reference only):")
    for k, v in raw_feature_ranges.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
