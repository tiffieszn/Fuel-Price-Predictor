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
def load_bundle():
    model_path = Path(__file__).resolve().parent / "model.pkl"
    if not model_path.exists():
        st.error("model.pkl not found in app directory. Run train_model.py and commit model.pkl.")
        st.stop()
    return joblib.load(model_path)

bundle = load_bundle()

# ---- SAFE extraction with fallbacks (no more KeyError) ----
model = bundle.get("model")
encoders = bundle.get("encoders", {})
features = bundle.get("features", ["open", "low", "close", "volume_log10", "commodity"])

ui_ranges_by_commodity = bundle.get("ui_ranges_by_commodity", {})  # new
ui_ranges_global = bundle.get("ui_ranges_global")                  # new

# fallback to legacy global ranges
if ui_ranges_global is None:
    ui_ranges_global = bundle.get("ui_feature_ranges")  # old key

if model is None or "commodity" not in encoders or ui_ranges_global is None:
    st.error(
        "Your model bundle is missing required artifacts. "
        "Please re-run train_model.py and redeploy with the latest model.pkl."
    )
    with st.expander("debug bundle keys"):
        st.write(sorted(bundle.keys()))
    st.stop()

# ===== UI (center, no sidebar) =====
st.subheader("Your inputs")

colA, colB, colC = st.columns([1.2, 1.2, 1.2], gap="large")

commodity_options = list(encoders["commodity"].classes_)
with colA:
    commodity = st.selectbox("Commodity", commodity_options)

ranges = ui_ranges_by_commodity.get(commodity, ui_ranges_global)

open_min, open_max = ranges["open"]
low_min, low_max = ranges["low"]
close_min, close_max = ranges["close"]
vlog_min, vlog_max = ranges.get("volume_log10", ui_ranges_global["volume_log10"])

with colB:
    open_p = st.slider("Open Price", float(open_min), float(open_max), float(open_min))
    low_p = st.slider("Low Price", float(low_min), float(low_max), float(low_min))
    close_p = st.slider("Close Price", float(close_min), float(close_max), float(close_min))

with colC:
    volume_log10 = st.slider("Volume (log scale)", float(vlog_min), float(vlog_max), float(vlog_min))
    volume_real = int(round(10 ** volume_log10))
    st.metric("Volume (approx)", f"{volume_real:,}")

# vibe checks (not blocking)
notes = []
if low_p > min(open_p, close_p):
    notes.append("Low is higher than Open/Close — kinda sus, but okay.")
if notes:
    st.warning(" ".join(notes))

# build model input
commodity_encoded = encoders["commodity"].transform([commodity])[0]
input_df = pd.DataFrame(
    [[open_p, low_p, close_p, volume_log10, commodity_encoded]],
    columns=features,
)

st.divider()

left, right = st.columns([1, 2], gap="large")
with left:
    predict = st.button("Predict 🔮", type="primary", use_container_width=True)
with right:
    st.caption("Ranges adapt per commodity, so the sliders won’t go full chaos.")

if predict:
    pred = float(model.predict(input_df)[0])

    st.subheader("Result")
    st.metric("Predicted High Price", f"{pred:,.2f}")

    with st.expander("Show input summary", expanded=True):
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": ["Commodity", "Open", "Low", "Close", "Volume (approx)"],
                    "Value": [commodity, open_p, low_p, close_p, volume_real],
                }
            ),
            use_container_width=True,
        )

    # feature importance
    st.subheader("What mattered most (feature importance)")
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        st.info("This model doesn’t expose feature importance.")
    else:
        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_xlabel("Importance")
