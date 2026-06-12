import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("crop_recommendation_model.joblib")

try:
    model_data = load_model()
    pipeline = model_data["pipeline"]
    label_encoder = model_data["label_encoder"]
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
background: linear-gradient(
135deg,
#0f172a 0%,
#14532d 50%,
#052e16 100%
);
}

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Hero */
.hero{
padding:25px;
border-radius:25px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(15px);
border:1px solid rgba(255,255,255,0.1);
margin-bottom:20px;
}

.hero-title{
font-size:55px;
font-weight:700;
color:white;
text-align:center;
}

.hero-sub{
font-size:18px;
text-align:center;
color:#cbd5e1;
}

/* Cards */
.card{
background:rgba(255,255,255,0.08);
padding:20px;
border-radius:20px;
backdrop-filter:blur(12px);
border:1px solid rgba(255,255,255,0.1);
}

/* Labels */
label{
color:white !important;
font-weight:600 !important;
}

/* Inputs */
.stNumberInput input{
background:#f8fafc !important;
color:#0f172a !important;
border-radius:12px !important;
font-size:18px !important;
font-weight:600 !important;
}

/* Button */
.stButton button{
width:100%;
height:60px;
border:none;
border-radius:15px;
background:linear-gradient(135deg,#22c55e,#16a34a);
color:white;
font-size:20px;
font-weight:700;
}

/* Result */
.result{
background:linear-gradient(135deg,#22c55e,#15803d);
padding:30px;
border-radius:20px;
text-align:center;
font-size:35px;
font-weight:700;
color:white;
margin-top:20px;
box-shadow:0 10px 25px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
<div class="hero-title">🌱 Smart Crop Recommendation</div>
<div class="hero-sub">
AI Powered Farming Assistant for Better Yield & Smart Agriculture
</div>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🌾 Soil & Weather Parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 90)

with col2:
    P = st.number_input("Phosphorus (P)", 0, 200, 42)

with col3:
    K = st.number_input("Potassium (K)", 0, 200, 43)

with col4:
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

col5, col6, col7 = st.columns(3)

with col5:
    temperature = st.number_input("Temperature (°C)", value=20.8)

with col6:
    humidity = st.number_input("Humidity (%)", value=82.0)

with col7:
    rainfall = st.number_input("Rainfall (mm)", value=202.0)

predict = st.button("🚀 Recommend Best Crop")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FEATURED CROPS ----------------
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("🌾 Popular Crops")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.image(
        "https://images.unsplash.com/photo-1536657464919-892534f60d6e",
        use_container_width=True
    )
    st.caption("Rice")

with c2:
    st.image(
        "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b",
        use_container_width=True
    )
    st.caption("Wheat")



with c3:
    st.image(
        "https://images.unsplash.com/photo-1464226184884-fa280b87c399",
        use_container_width=True
    )
    st.caption("Cotton")

# ---------------- PREDICTION ----------------
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

    st.markdown(
        f"""
        <div class="result">
        🌾 Recommended Crop<br><br>
        {crop.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )

   

