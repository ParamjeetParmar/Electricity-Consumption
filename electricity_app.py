import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# === Load model and data ===
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgb_electricity_model.pkl")
    return joblib.load(model_path)

@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/ParamjeetParmar/Electricity-Consumption/main/ICT_Subdimension_Dataset%20new%20(1).csv"
    df = pd.read_csv(csv_url)
    le = LabelEncoder()
    df['City_encoded'] = le.fit_transform(df['City'])
    return df, le

# === Setup Page ===
st.set_page_config(page_title="Electricity Forecast", layout="wide")

# === Custom CSS ===
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .reportview-container { padding-top: 1rem; }
        .block-container { padding: 2rem; }
        .big-font { font-size:30px !important; font-weight: bold; color: #ff7e00; }
        .sub-font { font-size:20px !important; color: #333333; }
        .dataframe th, .dataframe td { text-align: center; }
        .stButton>button { background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("<div class='big-font'>⚡ Electricity Consumption Forecast App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-font'>Predict the future of smart meter deployment across Indian cities</div>", unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.markdown("### 📅 Select Year Range")
start_year, end_year = st.sidebar.slider("Predict from year to year", 2025, 3025, (2030, 3025))
future_years = list(range(start_year, end_year + 1))

# === Load model and data ===
model = load_model()
df, le = load_data()

# === Prepare base data
latest_year = df["Year"].max()
base_df = df[df["Year"] == latest_year].copy()
model_features = model.get_booster().feature_names

# === Prediction logic
predictions = []
growth_rate = 0.03  # 3% growth

for i, year in enumerate(future_years, start=1):
    future_data = base_df.copy()
    future_data["Year"] = year
    for col in model_features:
        if col in future_data.columns and col not in ["Year", "City", "City_encoded"]:
            if pd.api.types.is_numeric_dtype(future_data[col]):
                future_data[col] *= (1 + growth_rate) ** i
    if "City" in model_features:
        future_data["City"] = future_data["City_encoded"]
    X = future_data[model_features].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
    y_pred = model.predict(X)
    future_data["Predicted Smart Meters (%)"] = y_pred
    future_data["Predicted Year"] = year
    future_data["City"] = le.inverse_transform(future_data["City_encoded"])
    predictions.append(future_data[["City", "Predicted Year", "Predicted Smart Meters (%)"]])

result_df = pd.concat(predictions, ignore_index=True)

# === Prediction Table ===
st.markdown("### 📊 Prediction Table")
with st.container():
    st.dataframe(result_df.style.format({"Predicted Smart Meter_
