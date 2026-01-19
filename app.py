import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Fuel Price Predictor", layout="wide")

st.title("Fuel Price Predictor")
st.write(
    "This application predicts the **highest daily fuel price** "
    "based on historical market features."
)

with st.expander("How to use", expanded=True):
    st.markdown(
        """
1. Select a fuel commodity.
2. Adjust the market parameters using the sliders.
3. Click **Predict** to estimate the price.

All input ranges are bounded based on the training data to ensure valid predictions.
"""
    )

@st.cache_resource
def load_artifacts():
    model_path = Path(__file__).resolve().parent / "model.pkl"
    bundle = joblib.load(model_path)

    return (
        bundle["model"],
        bundle["encoders"],
        bundle["feature_ranges"],
        bundle["features"],
    )


model, encoders, feature_ranges, feature_names = load_artifacts()

st.sidebar.header("Input Parameters")

commodity_options = list(encoders["commodity"].classes_)
commodity = st.sidebar.selectbox("Commodity", commodity_options)

open_min, open_max = feature_ranges["open"]
low_min, low_max = feature_ranges["low"]
close_min, close_max = feature_ranges["close"]
vol_min, vol_max = feature_ranges["volume"]

open_p = st.sidebar.slider("Open Price", open_min, open_max, open_min)
low_p = st.sidebar.slider("Low Price", low_min, low_max, low_min)
close_p = st.sidebar.slider("Close Price", close_min, close_max, close_min)
volume = st.sidebar.slider("Volume", vol_min, vol_max, vol_min)

warnings = []
if low_p > min(open_p, close_p):
    warnings.append("Low price is higher than Open/Close. Please check the inputs.")

if volume == 0:
    warnings.append("Volume is zero. Prediction reliability may be reduced.")

if warnings:
    st.sidebar.warning(" ".join(warnings))

commodity_encoded = encoders["commodity"].transform([commodity])[0]

input_df = pd.DataFrame(
    [[open_p, low_p, close_p, volume, commodity_encoded]],
    columns=feature_names,
)

predict_button = st.button("Predict", type="primary")

if predict_button:
    prediction = float(model.predict(input_df)[0])

    st.subheader("Prediction Result")
    st.metric("Predicted High Price", f"{prediction:,.2f}")

    st.subheader("Feature Importance")
    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    ax.set_xlabel("Importance")
    ax.set_title("Model Feature Importance")
    st.pyplot(fig)

    st.subheader("Input Summary")
    summary_df = pd.DataFrame(
        {
            "Feature": ["Commodity", "Open", "Low", "Close", "Volume"],
            "Value": [commodity, open_p, low_p, close_p, volume],
        }
    )
    st.dataframe(summary_df, use_container_width=True)

else:
    st.info("Adjust the inputs on the left and click **Predict** to see the result.")
