import streamlit as st
import joblib
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

.main-title{
    text-align:center;
    font-size:3rem;
    font-weight:700;
    color:white;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    color:#dbeafe;
    font-size:1.1rem;
    margin-bottom:30px;
}

.glass{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    border-radius:20px;
    padding:25px;
    border:1px solid rgba(255,255,255,0.15);
}

.result-card{
    background: linear-gradient(135deg,#00c853,#64dd17);
    padding:25px;
    border-radius:18px;
    text-align:center;
    color:white;
    font-size:28px;
    font-weight:700;
    box-shadow:0px 8px 25px rgba(0,0,0,0.25);
}

.metric-card{
    background: rgba(255,255,255,0.08);
    border-radius:16px;
    padding:20px;
    text-align:center;
    color:white;
}

div.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:15px;
    background:linear-gradient(90deg,#00c853,#00e676);
    color:white;
    font-size:18px;
    font-weight:600;
}

div.stButton > button:hover{
    transform:scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    "<div class='main-title'>🌱 Smart Crop Recommendation System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI Powered Crop Prediction using Soil & Weather Parameters</div>",
    unsafe_allow_html=True
)

# Load Model
try:
    model_data = joblib.load("crop_recommendation_model.joblib")
    pipeline = model_data["pipeline"]
    label_encoder = model_data["label_encoder"]
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("🌾 Enter Farm Details")

    N = st.slider("Nitrogen (N)", 0, 140, 90)
    P = st.slider("Phosphorus (P)", 0, 140, 42)
    K = st.slider("Potassium (K)", 0, 140, 43)

    temperature = st.number_input("🌡 Temperature (°C)", value=20.8)
    humidity = st.number_input("💧 Humidity (%)", value=82.0)
    ph = st.number_input("⚗ Soil pH", value=6.5)
    rainfall = st.number_input("🌧 Rainfall (mm)", value=202.0)

    predict = st.button("🚀 Recommend Best Crop")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)

    st.markdown("### 📊 Smart Agriculture")

    st.markdown("""
    ✔ Soil Analysis

    ✔ Weather-Based Prediction

    ✔ AI Recommendation

    ✔ Sustainable Farming
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# Prediction
if predict:
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

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='result-card'>🌾 Recommended Crop<br><br>{crop.upper()}</div>",
        unsafe_allow_html=True
    )

    st.balloons()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    "<center style='color:white;'>Built with ❤️ using Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)
