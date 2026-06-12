import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
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
    st.error(f"Model Loading Error: {e}")
    st.stop()

# ---------------- CSS ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
background-image:url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=1600&q=80");
background-size:cover;
background-position:center;
background-attachment:fixed;
}

.main-container{
background:rgba(0,0,0,0.55);
padding:30px;
border-radius:25px;
backdrop-filter:blur(12px);
}

.title{
font-size:58px;
font-weight:700;
text-align:center;
color:white;
}

.subtitle{
font-size:18px;
text-align:center;
color:#f1f5f9;
margin-bottom:25px;
}

.result{
background:linear-gradient(135deg,#22c55e,#15803d);
padding:25px;
border-radius:20px;
text-align:center;
font-size:32px;
font-weight:700;
color:white;
margin-top:25px;
box-shadow:0 10px 25px rgba(0,0,0,0.3);
}

.crop-card{
background:rgba(255,255,255,0.1);
padding:15px;
border-radius:15px;
backdrop-filter:blur(10px);
}

h1,h2,h3,label{
color:white !important;
}

div.stButton > button{
width:100%;
height:60px;
border:none;
border-radius:15px;
background:linear-gradient(90deg,#16a34a,#22c55e);
color:white;
font-size:20px;
font-weight:600;
}

div.stButton > button:hover{
transform:scale(1.02);
}

[data-testid="stNumberInput"] label{
font-weight:600;
color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="main-container">
<div class="title">🌱 Smart Crop Recommendation</div>
<div class="subtitle">
AI-Based Crop Prediction Using Soil Nutrients & Weather Conditions
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- INPUTS ----------------
st.markdown("## 🌾 Enter Farm Details")

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

# ---------------- FEATURED CROPS ----------------
st.markdown("---")
st.markdown("## 🌾 Featured Crops")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.image(
        "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?auto=format&fit=crop&w=500&q=80",
        use_container_width=True
    )
    st.markdown("### Rice")

with c2:
    st.image(
        "https://images.unsplash.com/photo-1601597111158-2fceff292cdc?auto=format&fit=crop&w=500&q=80",
        use_container_width=True
    )
    st.markdown("### Wheat")

with c3:
    st.image(
        "https://images.unsplash.com/photo-1502741338009-cac2772e18bc?auto=format&fit=crop&w=500&q=80",
        use_container_width=True
    )
    st.markdown("### Maize")

with c4:
    st.image(
        "https://images.unsplash.com/photo-1592928302636-c83cf1e1f6f3?auto=format&fit=crop&w=500&q=80",
        use_container_width=True
    )
    st.markdown("### Cotton")

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

    st.balloons()

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<center style='color:white;font-size:16px;'>🚜 Built with Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)
