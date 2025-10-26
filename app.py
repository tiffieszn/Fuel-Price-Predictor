import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Fuel Price Predictor", layout='wide')
st.title("Fuel Price Prediction App")
st.write("Predict the **highest daily fuel price ('high')** based on market data.")

#load model
@st.cache_resource
def load_model():
    data = joblib.load('model.pkl')
    return data['model'], data['encoders']
model, encoders = load_model()

st.sidebar.header('input parametes')
def user_input_features():
    open_p = st.sidebar.number_input('open price', value=30.0)
    low_p = st.sidebar.number_input('low price', value=29.5)
    close_p = st.sidebar.number_input('close price', value=30.2)
    volume = st.sidebar.number_input('volume', value=50000)
    commodity = st.sidebar.text_input('commodity', 'crude oil')
    ticker = st.sidebar.text_input('Ticker', 'CL=F')

    data = pd.DataFrame({
        'open': [open_p],
        'low': [low_p],
        'close': [close_p],
        'volume': [volume],
        'commodity': [commodity],
        'ticker': [ticker]
    })
    return data

input_df = user_input_features()

for col in ['commodity', 'ticker']:
    le = encoders[col]
    try:
        input_df[col] = le.transform(input_df[col])
    except ValueError:
        st.warning(f"'{input_df[col].values[0]}' not in training data for {col}. using default 0.")
        input_df[col] = 0

prediction = model.predict(input_df)[0]
st.metric("Predicted High Price", f"{prediction:.2f}")
st.subheader("Feature Importance")
importances = model.feature_importances_
features = ['open', 'low', 'close', 'volume', 'commodity', 'ticker']

fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_xlabel('importance')
ax.set_title('feature importance')
st.pyplot(fig)

st.subheader('Sample Prediction Distribution')
sample_pred = np.random.normal(prediction, 0.2, 50)
plt.figure(figsize=(6, 3))
plt.hist(sample_pred, bins=25, alpha=0.7)
st.pyplot(plt)