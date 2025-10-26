import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("all_fuels_data.csv")

features = ["open", "low", "close", "volume", 'commodity', "ticker"]
target = "high"

encoders = {}
for col in ['commodity', 'ticker']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

x = df[features]
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

joblib.dump({"model": model, "encoders": encoders}, "model.pkl")
print("Model trained and saved as model.pkl")