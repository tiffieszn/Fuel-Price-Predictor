import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Fuel Price Predictor", layout="wide")

st.title("Fuel Price Predictor ⛽📈")
st.caption("Pick a commodity, slide the knobs, get the predicted **High** price. No drama.")

with st.expander("what even is this?", expanded=False):
    st.markdown(
        """
This app predicts the **highest daily fuel price (High)** using a trained regression model.

**How to use (speedrun):**
1) Choose a commodity  
2) Adjust Open / Low / Close + Volume  
3) Hit **Predict**  
"""
    )

@st.cache_resource
def load_artifacts():
    model_path = Path(__file__).resolve().parent / "model.pkl"
    if not model_path.exists():
        st.error(
            "model.pkl not found. Run `python train_model.py` locally, commit `model.pkl`, then redeploy."
        )
        st.stop()

    bundle = joblib.load(model_path)
    return bundle

bundle = load_artifacts()

model = bundle["model"]
encoders = bundle["encoders"]
features = bundle["features"]  # ["open","low","close","volume_log10","commodity"]
ui_ranges_by_commodity = bundle["ui_ranges_by_commodity"]
ui_ranges_global = bundle["ui_ranges_global"]

st.subheader("Your inputs")

colA, colB, colC = st.columns([1.2, 1.2, 1.2], gap="large")

with colA:
    commodity_options = list(encoders["commodity"].classes_)
    commodity = st.selectbox("Commodity", commodity_options)

ranges = ui_ranges_by_commodity.get(commodity, ui_ranges_global)

open_min, open_max = ranges["open"]
low_min, low_max = ranges["low"]
close_min, close_max = ranges["close"]
vlog_min, vlog_max = ranges["volume_log10"]

with colB:
    open_p = st.slider("Open Price", open_min, open_max, open_min)
    low_p = st.slider("Low Price", low_min, low_max, low_min)
    close_p = st.slider("Close Price", close_min, close_max, close_min)

with colC:
    volume_log10 = st.slider("Volume (log scale)", vlog_min, vlog_max, vlog_min)
    volume_real = int(round(10 ** volume_log10))
    st.metric("Volume (approx)", f"{volume_real:,}")

notes = []
if low_p > min(open_p, close_p):
    notes.append("Low is higher than Open/Close. That’s kinda sus, but I’ll still predict.")

if notes:
    st.warning(" ".join(notes))

commodity_encoded = encoders["commodity"].transform([commodity])[0]

input_df = pd.DataFrame(
    [[open_p, low_p, close_p, volume_log10, commodity_encoded]],
    columns=features,
)

st.divider()

cta_left, cta_right = st.columns([1, 2], gap="large")

with cta_left:
    predict_button = st.button("Predict 🔮", type="primary", use_container_width=True)

with cta_right:
    st.caption("Tip: ranges adapt per commodity, so the sliders won’t go full chaos.")

if predict_button:
    pred = float(model.predict(input_df)[0])

    st.subheader("Result")
    st.metric("Predicted High Price", f"{pred:,.2f}")

    with st.expander("Show input summary", expanded=True):
        summary_df = pd.DataFrame(
            {
                "Feature": ["Commodity", "Open", "Low", "Close", "Volume (approx)"],
                "Value": [commodity, open_p, low_p, close_p, volume_real],
            }
        )
        st.dataframe(summary_df, use_container_width=True)

    st.subheader("What mattered most (feature importance)")
    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        st.info("This model doesn’t expose feature importance.")
    else:
        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

else:
    st.info("Slide the inputs above, then smash **Predict 🔮**.")
