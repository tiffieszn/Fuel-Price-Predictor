# Fuel Price Predictor

Fuel Price Predictor is a machine learning application designed to predict the **highest daily fuel price (High)** based on historical market data.

This project was developed as part of the **MLOps course (COMP6984001)** and demonstrates a complete workflow from **model training to deployment** using **Streamlit**.

---

## Project Overview

Fuel price volatility plays a critical role in energy markets, logistics planning, and economic decision-making.

This project formulates fuel price prediction as a **supervised regression problem**, focusing on estimating the highest daily price using historical price and volume data across multiple fuel commodities.

The system consists of:
- An **offline training script** for model building and serialization
- A **Streamlit web application** for inference and visualization
- A **clean repository structure** aligned with MLOps best practices

---

## Features

- Predicts **highest daily fuel price (High)**
- Supports multiple fuel commodities:
  - Crude Oil
  - Brent Crude Oil
  - Natural Gas
  - Heating Oil
  - RBOB Gasoline
- Streamlit interface with:
  - Slider-based numerical inputs bounded by training data ranges
  - Dropdown selection for categorical variables
- Feature importance visualization
- Clear separation between **training** and **inference**

---

## Repository Structure

```text
Fuel-Price-Predictor/
│
├── app.py                 # Streamlit inference application
├── train_model.py         # Model training script
├── model.pkl              # Trained model and artifacts (generated after training)
├── all_fuels_data.csv     # Dataset
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
└── .github/workflows/     # CI/CD pipeline configuration
```

---

## Dataset

- **Source**: Open historical fuel market data
- **File**: `all_fuels_data.csv`

### Features
- `open`
- `low`
- `close`
- `volume`
- `commodity`

### Target
- `high`

---

## Model

- **Algorithm**: Gradient Boosting Regressor
- **Objective**: Squared error loss

### Key Hyperparameters
- `learning_rate = 0.05`
- `n_estimators = 500`
- `max_depth = 5`

### Preprocessing
- Commodity values are encoded using **Label Encoding**
- Feature ranges are extracted from training data and stored for safe inference

---

## Training the Model

Run the training script to generate the trained model and artifacts:

```bash
python train_model.py
```

This will produce `model.pkl`, which contains:
- Trained model
- Encoders
- Feature ranges
- Feature metadata

> Training is performed **offline** and should not be repeated inside the Streamlit application.

---

## Running the Streamlit App

After training the model, launch the application locally:

```bash
streamlit run app.py
```

The application will:
- Load the pre-trained model from `model.pkl`
- Accept user inputs through sliders and dropdowns
- Perform inference
- Display prediction results and visualizations

---

## Application Workflow

1. User selects a fuel commodity
2. User adjusts market parameters using sliders
3. Input values are validated against training data ranges
4. The model predicts the highest daily price
5. Results and feature importance are displayed

---

## MLOps Practices Applied

- Clear separation between training and inference
- Serialized model artifacts
- Reproducible environment via `requirements.txt`
- Clean repository structure
- CI/CD pipeline for automated checks

---

## Requirements

### Main Dependencies
- Python 3.9+
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Demo Video

A short demo video (≤ 1 minute) is provided separately, demonstrating:
- Application usage
- End-to-end workflow

---

## Team Members

- **Muhammad Iqbal Saputra** – 2702390236  
- **Tiffanny Rosyanna Dewi** – 2802508666  
- **Meyathala Razditya** – 2802535673  
- **Ellen Ardelia Hartono** – 2802513685  

**Program**: Bachelor of Artificial Intelligence  
**Institution**: Bina Nusantara University  
**Course Code**: COMP6984001  
**Semester**: Odd
