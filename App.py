import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai
from PIL import Image

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="Elite-Racing-Agent PRO", page_icon="🏁", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    [data-testid="stMetricValue"] { font-size: 42px !important; font-family: 'Courier New'; color: #00ff41; }
    .status-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 20px; }
    .stButton>button { height: 3em; background-color: #00ff41; color: black; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- API & SESSION STATE ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕹️ COMMAND CENTER")
    weight = st.number_input("Mass (kg)", value=850)
    power = st.number_input("Power (HP)", value=600)
    cd = st.slider("Aero Drag (Cd)", 0.1, 0.9, 0.45)
    mu = st.slider("Tire Grip (Coefficient)", 0.5, 2.0, 1.2) # Added for realistic graph clipping
    track_input = st.text_input("Track", value="Strawberry Creek Raceway")
    
    if st.button("SYNC LIVE WEATHER"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat=53.3377&lon=-114.1603&appid={WEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            temp_k = res['main']['temp'] + 273.15
            press_pa = res['main']['pressure'] * 100
            st.session_state['rho'] = round(press_pa / (287.05 * temp_k), 4)
        except: st.error("Sync Failed")

rho = st.session_state['rho']

# --- MAIN DASHBOARD ---
st.title("🏁 ELITE-RACING-AGENT | PRO-SERIES")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Performance Map: Target vs Actual")
    
    # --- CORRECTED PHYSICS ENGINE ---
    v_kmh = np.linspace(5, 250, 100)
    v_ms = v_kmh / 3.6
    power_w = power * 745.7
    
    # Calculate acceleration and clip it at the tire's traction limit (mu)
    # This keeps the line from disappearing off the top of the chart
    a_g = []
    for v in v_ms:
        accel_raw = ( (power_w / v) - (0.5 * rho * v**2 * cd * 1.5) ) / (weight * 9.81)
        a_g.append(min(accel_raw, mu)) # Traction Limit Clipping
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    ax.plot(v_kmh, a_g, color='#00ff41', linewidth=4, label="DIGITAL TWIN LIMIT")
    
    uploaded_file = st.file_uploader("Upload Telemetry CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'speed' in df.columns and 'accel' in df.columns:
            ax.scatter(df['speed'], df['accel'], color='#ff4b4b', s=15, alpha=0.6, label="ACTUAL")
    
    ax.set_ylim(0, 1.6) # Lock y-axis so it's readable
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Longitudinal G")
    ax.grid(True, alpha=0.1)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("G-G Friction Circle")
    fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
    # Reference Circle (The limit of the tires)
    circle = plt.Circle((0, 0), mu, color='#00ff41', fill=False, linestyle='--', linewidth=2)
    ax_gg.add_artist(circle)
    
    if uploaded_file and 'lat_g' in df.columns and 'accel' in df.columns:
        ax_gg.scatter(df['lat_g'], df['accel'], color='white', s=5, alpha=0.4)
    
    ax_gg.set_xlim(-1.6, 1.6)
    ax_gg.set_ylim(-1.6, 1.6)
    ax_gg.axhline(0, color='grey', lw=1, alpha=0.5)
    ax_gg.axvline(0, color='grey', lw=1, alpha=0.5)
    ax_gg.set_xlabel("Lateral G")
    ax_gg.set_ylabel("Longitudinal G")
    st.pyplot(fig_gg)

# Metrics Footer
m1, m2, m3 = st.columns(3)
m1.metric("AIR DENSITY", f"{rho} kg/m³")
m2.metric("PWR/WGT", f"{round(power/weight, 3)}")
m3.metric("TRACK", track_input)
