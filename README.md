# Fuel-Price-Predictor
A machine learning web application built with **Streamlit** that predicts the **highest daily fuel price (`high`)** based on market features such as open, low, close, volume, and commodity type.

The app also provides:
- Feature importance visualization  
- Sample prediction distribution  


# Project Overview
This project consists of two main components:

1. **Model Training (`train_model.py`)**
   - Trains a machine learning model using historical fuel price data.
   - Uses **Random Forest Regressor**.
   - Encodes categorical variables (`commodity`).
   - Saves the trained model and encoders as `model.pkl`.

2. **Web Application (`app.py`)**
   - Built with **Streamlit**.
   - Loads the trained model.
   - Allows users to input values.
   - Generates predictions and visualizations only after clicking **“Predict Fuel Price”**.


# Dataset
The model is trained using:
all_fuels_data.csv

Required columns:
- `open`
- `low`
- `close`
- `volume`
- `commodity`
- `high` (target variable)


# Machine Learning Algorithm
We use:
**Random Forest Regressor**

Why this model?
- Handles non-linear relationships well  
- Robust to outliers  
- Provides feature importance  
- Works well with mixed numerical and encoded categorical data  

Key parameters:
- `n_estimators = 150`
- `random_state = 42`
- `n_jobs = -1` (uses all CPU cores)


# Training Process (Summary)
1. Load dataset (`all_fuels_data.csv`)
2. Encode categorical column (`commodity`) using LabelEncoder  
3. Define features (`X`) and target (`y = high`)  
4. Train Random Forest model  
5. Save model and encoders as `model.pkl`


# How to Run Locally
1. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate   # Windows

2. Install dependencies
pip install -r requirements.txt

3. Train the Model
python train_model.py

4. Run the streamlit app
streamlit run app.py


