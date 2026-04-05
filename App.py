import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai
from PIL import Image

# --- CONFIGURATION & THEME ---
st.set_page_config(
    page_title="Elite-Racing-Agent",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force Dark Mode Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; background-color: #00ff41; color: black; font-weight: bold; }
    [data-testid="stMetricValue"] { color: #00ff41; }
    </style>
    """, unsafe_allow_html=True)

# --- API SETUP ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- CORE PHYSICS ENGINE ---
def calculate_digital_twin(power_hp, weight_kg, cd, rho):
    """Calculates theoretical G-force curve vs Speed."""
    power_watts = power_hp * 745.7
    speeds_kmh = np.linspace(5, 280, 100) 
    speeds_ms = speeds_kmh / 3.6
    frontal_area = 1.5 
    
    accel_g = []
    for v in speeds_ms:
        # F_net = (Power/v) - (0.5 * rho * v^2 * Cd * A)
        force_engine = power_watts / v
        force_drag = 0.5 * rho * (v**2) * cd * frontal_area
        net_accel_ms2 = (force_engine - force_drag) / weight_kg
        accel_g.append(max(0, net_accel_ms2 / 9.81))
            
    return speeds_kmh, accel_g

# --- UI LAYOUT ---
st.title("🏎️ ELITE-RACING-AGENT | Championship OS")
st.markdown("---")

# Sidebar for Inputs
with st.sidebar:
    st.header("Vehicle DNA")
    weight = st.number_input("Total Weight (kg)", value=850)
    power = st.number_input("Engine Power (HP)", value=600)
    drag_coeff = st.slider("Aero Drag (Cd)", 0.2, 0.9, 0.45)
    
    st.header("Environment")
    city = st.text_input("Track City", "Colorado Springs")
    
    # Dynamic Air Density Calculation
    if st.button("Sync Live Weather"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            temp_k = res['main']['temp'] + 273.15
            press_pa = res['main']['pressure'] * 100
            # rho = P / (R * T)
            st.session_state['rho'] = round(press_pa / (287.05 * temp_k), 4)
            st.success(f"Synced: {res['main']['temp']}°C")
        except:
            st.error("Weather Sync Failed. Using Sea Level.")
            st.session_state['rho'] = 1.225

    current_rho = st.session_state.get('rho', 1.225)
    st.metric("Air Density (ρ)", f"{current_rho} kg/m³")

# Main Dashboard Grid
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Performance Map: Digital Twin vs. Real World")
    v_kmh, a_g = calculate_digital_twin(power, weight, drag_coeff, current_rho)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    ax.plot(v_kmh, a_g, color='#00ff41', linewidth=3, label="Digital Twin (Target)")
    
    uploaded_file = st.file_uploader("Upload Telemetry CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'speed' in df.columns and 'accel' in df.columns:
            ax.scatter(df['speed'], df['accel'], color='#ff4b4b', s=12, label="Live Telemetry")
    
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Longitudinal G")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Strategy & Insights")
    try:
        st.image(Image.open("1000006405.png"), use_container_width=True)
    except:
        st.warning("Upload '1000006405.png' to GitHub.")

    # Dynamic Probability Logic
    p_to_w = power / weight
    # Penalty based on density deviation from sea level
    density_ratio = current_rho / 1.225
    prob_score = min(99, max(10, int((p_to_w * density_ratio / 0.75) * 100)))
    
    st.metric("Winning Probability", f"{prob_score}%", f"{round(current_rho - 1.225, 2)} ρ-Delta")

    if GOOGLE_API_KEY:
        if st.button("Generate Strategy Brief"):
            model = genai.GenerativeModel('gemini-flash-latest')
            prompt = f"You are a Race Engineer. Car: {power}HP, {weight}kg. Track: {city} at {current_rho} density. Prob: {prob_score}%. Give 2 tactical tips."
            st.info(model.generate_content(prompt).text)

st.markdown("---")
st.caption("Elite-Racing-Agent v1.1 | Developed for High-Performance Validation")
