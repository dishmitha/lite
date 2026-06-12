import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Crop Recommendation", page_icon="🌱")

st.title("🌱 Crop Recommendation System")

# Load model
model_data = joblib.load("crop_recommendation_model.joblib")

pipeline = model_data["pipeline"]
label_encoder = model_data["label_encoder"]
feature_names = model_data["feature_names"]

st.subheader("Enter Soil and Weather Details")

N = st.number_input("Nitrogen (N)", 0, 200, 90)
P = st.number_input("Phosphorus (P)", 0, 200, 42)
K = st.number_input("Potassium (K)", 0, 200, 43)

temperature = st.number_input("Temperature (°C)", value=20.8)
humidity = st.number_input("Humidity (%)", value=82.0)
ph = st.number_input("pH Value", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=202.0)

if st.button("Recommend Crop"):
    sample = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    prediction = pipeline.predict(sample)
    crop = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Recommended Crop: 🌾 {crop}")
