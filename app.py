import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Crop Recommendation", page_icon="🌱")

st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0b3d20 0%,
        #14532d 40%,
        #1f7a3f 100%
    );
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Title */
.main-title{
    text-align:center;
    color:white;
    font-size:3.5rem;
    font-weight:700;
    margin-bottom:10px;
}

.sub-title{
    text-align:center;
    color:#d1fae5;
    font-size:1.1rem;
    margin-bottom:30px;
}

/* Input Labels */
label{
    color:white !important;
    font-weight:600 !important;
    font-size:16px !important;
}

/* Input Boxes */
.stNumberInput input{
    background:#ffffff !important;
    color:#14532d !important;
    border-radius:12px !important;
    border:2px solid #22c55e !important;
    font-size:18px !important;
    font-weight:600 !important;
}

/* Subheader */
h3{
    color:white !important;
}

/* Button */
.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:15px;
    background:linear-gradient(
        135deg,
        #22c55e,
        #16a34a
    );
    color:white;
    font-size:18px;
    font-weight:700;
    transition:0.3s;
}

.stButton > button:hover{
    transform:scale(1.02);
}

/* Success Box */
[data-testid="stSuccess"]{
    border-radius:15px;
    font-size:18px;
    font-weight:600;
}

/* Card Effect */
.block-container{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(12px);
    padding:2rem;
    border-radius:25px;
    margin-top:2rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
🌾 Smart Crop Recommendation System
</div>

<div class="sub-title">
AI-Powered Farming Assistant for Better Yield & Smart Agriculture
</div>
""", unsafe_allow_html=True)

st.markdown("")

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
