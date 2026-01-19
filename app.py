import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

st.set_page_config(page_title="Fuel Price Predictor", layout="wide")

st.title("Fuel Price Predictor")
st.caption("This app predicts the highest daily fuel price (High) based on historical market features.")

with st.expander("About this app", expanded=False):
    st.markdown(
        """
**What it does**  
Predicts the **highest daily fuel price (High)** using a trained regression model.

**How to use**
1. Choose a commodity
2. Adjust Open / Low / Close and Volume
3. Click **Predict**
"""
    )


@st.cache_resource
def load_bundle():
    model_path = Path(__file__).resolve().parent / "model.pkl"
    if not model_path.exists():
        st.error("model.pkl was not found. Please run train_model.py and ensure model.pkl is committed.")
        st.stop()
    return joblib.load(model_path)


bundle = load_bundle()

model = bundle.get("model")
encoders = bundle.get("encoders", {})
features = bundle.get("features", ["open", "low", "close", "volume_log10", "commodity"])

ui_ranges_by_commodity = bundle.get("ui_ranges_by_commodity", {})
ui_ranges_global = bundle.get("ui_ranges_global") or bundle.get("ui_feature_ranges")

feature_importances = bundle.get("feature_importances")

if model is None or "commodity" not in encoders or ui_ranges_global is None:
    st.error("The model bundle is missing required artifacts. Please retrain and regenerate model.pkl.")
    with st.expander("Debug: bundle keys"):
        st.write(sorted(bundle.keys()))
    st.stop()

commodity_options = list(encoders["commodity"].classes_)

st.subheader("Input Parameters")

col1, col2, col3 = st.columns([1.2, 1.2, 1.2], gap="large")

with col1:
    commodity = st.selectbox("Commodity", commodity_options)

# Choose ranges (per commodity if available, otherwise global)
ranges = ui_ranges_by_commodity.get(commodity, ui_ranges_global)

open_min, open_max = ranges["open"]
low_min, low_max = ranges["low"]
close_min, close_max = ranges["close"]
vlog_min, vlog_max = ranges.get("volume_log10", ui_ranges_global["volume_log10"])

with col2:
    open_p = st.slider("Open Price", float(open_min), float(open_max), float(open_min))
    low_p = st.slider("Low Price", float(low_min), float(low_max), float(low_min))
    close_p = st.slider("Close Price", float(close_min), float(close_max), float(close_min))

with col3:
    volume_log10 = st.slider("Volume (log scale)", float(vlog_min), float(vlog_max), float(vlog_min))
    volume_real = int(round(10 ** volume_log10))
    st.metric("Volume (approx.)", f"{volume_real:,}")

warnings = []
if low_p > min(open_p, close_p):
    warnings.append("Low price is higher than Open/Close. Please check the inputs.")

if warnings:
    st.warning(" ".join(warnings))

commodity_encoded = encoders["commodity"].transform([commodity])[0]

input_df = pd.DataFrame(
    [[open_p, low_p, close_p, volume_log10, commodity_encoded]],
    columns=features,
)

st.divider()

predict = st.button("Predict", type="primary", use_container_width=True)

if predict:
    prediction = float(model.predict(input_df)[0])

    st.subheader("Result")
    st.metric("Predicted High Price", f"{prediction:,.2f}")

    with st.expander("Show input summary", expanded=True):
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": ["Commodity", "Open", "Low", "Close", "Volume (approx.)"],
                    "Value": [commodity, open_p, low_p, close_p, volume_real],
                }
            ),
            use_container_width=True,
        )

    st.subheader("Feature Importance")
    if feature_importances is None:
        st.info("Feature importance is not available for this simulations.")
    else:
        fig, ax = plt.subplots()
        ax.barh(features, feature_importances)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

else:
    st.info("Adjust the inputs above, then click Predict.")
