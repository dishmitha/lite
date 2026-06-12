import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

# ================= STYLING =================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* Beautiful Agriculture Background */
.stApp{
background:
radial-gradient(circle at top left, rgba(34,197,94,0.25), transparent 30%),
radial-gradient(circle at bottom right, rgba(132,204,22,0.20), transparent 30%),
linear-gradient(
135deg,
#041c12 0%,
#0b3d20 25%,
#14532d 60%,
#052e16 100%
);
}

/* Hide Streamlit UI */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Glass Card */
.block-container{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(20px);
padding:2rem;
border-radius:30px;
border:1px solid rgba(255,255,255,0.12);
margin-top:30px;
box-shadow:0 8px 30px rgba(0,0,0,0.25);
}

/* Title */
.main-title{
text-align:center;
font-size:3.5rem;
font-weight:700;
color:white;
margin-bottom:10px;
}

.sub-title{
text-align:center;
color:#bbf7d0;
font-size:18px;
margin-bottom:30px;
}

/* Labels */
label{
color:white !important;
font-weight:600 !important;
font-size:16px !important;
}

/* Inputs */
.stNumberInput input{
background:white !important;
color:#14532d !important;
border-radius:15px !important;
border:2px solid #22c55e !important;
font-size:18px !important;
font-weight:600 !important;
}

/* Section Header */
h3{
color:white !important;
}

/* Button */
.stButton > button{
width:100%;
height:60px;
border:none;
border-radius:18px;
background:linear-gradient(135deg,#22c55e,#16a34a);
color:white;
font-size:20px;
font-weight:700;
box-shadow:0 4px 20px rgba(34,197,94,0.4);
transition:0.3s;
}

.stButton > button:hover{
transform:translateY(-2px);
}

/* Success Result */
[data-testid="stSuccess"]{
background:rgba(34,197,94,0.2);
border:1px solid #22c55e;
border-radius:15px;
padding:15px;
font-size:20px;
font-weight:600;
}

/* Subheader Text */
.stMarkdown, .stText{
color:white;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="main-title">
🌾 Smart Crop Recommendation
</div>

<div class="sub-title">
AI-Powered Farming Assistant for Better Yield & Smart Agriculture
</div>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model_data = joblib.load("crop_recommendation_model.joblib")

pipeline = model_data["pipeline"]
label_encoder = model_data["label_encoder"]

# ================= INPUTS =================
st.subheader("🌱 Enter Soil and Weather Details")

N = st.number_input("Nitrogen (N)", 0, 200, 90)
P = st.number_input("Phosphorus (P)", 0, 200, 42)
K = st.number_input("Potassium (K)", 0, 200, 43)

temperature = st.number_input("Temperature (°C)", value=20.8)
humidity = st.number_input("Humidity (%)", value=82.0)
ph = st.number_input("pH Value", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=202.0)

# ================= PREDICTION =================
if st.button("🚀 Recommend Crop"):
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

    st.success(f"🌾 Recommended Crop: {crop.upper()}")

   
