import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("all_fuels_data.csv")

df = df.dropna(subset=["open", "low", "close", "volume", "commodity", "high"]).copy()

features = ["open", "low", "close", "volume", "commodity"]
target = "high"

encoders = {}
le = LabelEncoder()
df["commodity"] = le.fit_transform(df["commodity"])
encoders["commodity"] = le

feature_ranges = {
    "open": (float(df["open"].min()), float(df["open"].max())),
    "low": (float(df["low"].min()), float(df["low"].max())),
    "close": (float(df["close"].min()), float(df["close"].max())),
    "volume": (int(df["volume"].min()), int(df["volume"].max())),
}

X = df[features]
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    n_estimators=500,
    max_depth=5,
    random_state=42,
    max_features=None
)
model.fit(x_train, y_train)

bundle = {
    "model": model,
    "encoders": encoders,
    "feature_ranges": feature_ranges,
    "features": features,
    "target": target
}

joblib.dump(bundle, "model.pkl")
print("Model trained and saved as model.pkl")
print("Saved feature ranges for Streamlit sliders:", feature_ranges)
