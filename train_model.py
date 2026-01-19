import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("all_fuels_data.csv")

features = ["open", "low", "close", "volume", 'commodity']
target = "high"

encoders = {}
for col in ['commodity']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

x = df[features]
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.05, n_estimators=500, max_depth=5, random_state=42, max_features=None)
model.fit(x_train, y_train)

joblib.dump({"model": model, "encoders": encoders}, "model.pkl")
print("Model trained and saved as model.pkl")