import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai
from PIL import Image

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="Elite-Racing-Agent PRO", page_icon="🏁", layout="wide")

# Enhanced Pro-Racing CSS (Dark Mode & High Vis Metrics)
st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    [data-testid="stMetricValue"] { font-size: 42px !important; font-family: 'Courier New'; color: #00ff41; }
    .status-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 20px; }
    .stButton>button { height: 3em; background-color: #00ff41; color: black; border: none; }
    .stTextInput>div>div>input { background-color: #1e1e1e; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- API & SESSION STATE INITIALIZATION ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

if 'rho' not in st.session_state:
    st.session_state['rho'] = 1.225
if 'last_synced_city' not in st.session_state:
    st.session_state['last_synced_city'] = "None"

# --- SIDEBAR: COMMAND CENTER ---
with st.sidebar:
    st.title("🕹️ COMMAND CENTER")
    driver_mode = st.toggle("COCKPIT MODE (High Vis)", value=False)
    
    st.header("Vehicle DNA")
    weight = st.number_input("Mass (kg)", value=850)
    power = st.number_input("Power (HP)", value=600)
    cd = st.slider("Aero Drag (Cd)", 0.2, 0.9, 0.45)
    
    st.header("Environment")
    # Dynamic Location Input (Unlocked)
    track_input = st.text_input("Track or City Name", value="Strawberry Creek Raceway")
    
    if st.button("SYNC LIVE WEATHER"):
        try:
            # Special Coordinate Mapping for Strawberry Creek
            if "Strawberry Creek" in track_input:
                url = f"http://api.openweathermap.org/data/2.5/weather?lat=53.3377&lon=-114.1603&appid={WEATHER_API_KEY}&units=metric"
            else:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={track_input}&appid={WEATHER_API_KEY}&units=metric"
            
            res = requests.get(url).json()
            if res.get('cod') != 200:
                st.error(f"Error: {res.get('message', 'Unknown Error')}")
            else:
                temp_k = res['main']['temp'] + 273.15
                press_pa = res['main']['pressure'] * 100
                st.session_state['rho'] = round(press_pa / (287.05 * temp_k), 4)
                st.session_state['last_synced_city'] = track_input
                st.success(f"Synced: {res['main']['temp']}°C")
        except Exception as e:
            st.error("Sync Failed. Check API Key.")

    # Manual Override for Remote Tracks
    rho = st.slider("Air Density (ρ) Override", 0.800, 1.300, float(st.session_state['rho']), step=0.001)
    st.caption(f"Last Sync: {st.session_state['last_synced_city']}")

# --- CORE CALCULATIONS ---
p_to_w = power / weight
density_ratio = rho / 1.225
# Probability logic scaled for high-performance benchmarks
prob = min(99, max(10, int((p_to_w * density_ratio / 0.75) * 100)))

# Pit-Wall Status Banner Logic
if prob > 90: status, color, msg = "🟢 SEND IT", "#00ff41", "Optimal Power-to-Weight. Grip levels high."
elif prob > 75: status, color, msg = "🟡 CAUTION", "#ffff00", "Atmospheric drag increasing. Manage temps."
else: status, color, msg = "🔴 BOX BOX", "#ff4b4b", "Critical Density Altitude. Power output restricted."

# --- MAIN DASHBOARD ---
if not driver_mode:
    st.title("🏁 ELITE-RACING-AGENT | PRO-SERIES")
else:
    st.title("⏱️ LIVE TELEMETRY")

st.markdown(f'<div class="status-box" style="background-color: {color}; color: black;">{status} | {msg}</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # 1. Performance Map (The Digital Twin)
    st.subheader("Performance Map: Target vs Actual")
    v_kmh = np.linspace(10, 250, 100)
    # Physics formula: Accel = (Power/v - Drag) / Mass
    a_g = ( (power*745.7)/(v_kmh/3.6) - 0.5*rho*(v_kmh/3.6)**2*cd*1.5 ) / (weight*9.81)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    ax.plot(v_kmh, a_g, color='#00ff41', alpha=0.9, linewidth=4, label="DIGITAL TWIN LIMIT")
    
    uploaded_file = st.file_uploader("Drop Telemetry CSV (Requires 'speed', 'accel')", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'speed' in df.columns and 'accel' in df.columns:
            # Scatter plot color-coded by acceleration intensity
            ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='viridis', s=15, label="ACTUAL DATA")
    
    ax.set_ylabel("Longitudinal G-Force")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylim(0, 1.5)
    ax.legend()
    ax.grid(True, alpha=0.1)
    st.pyplot(fig)

with col2:
    # 2. G-G Friction Circle
    st.subheader("G-G Friction Circle")
    fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
    # Draw limit circle at 1.2G
    circle = plt.Circle((0, 0), 1.2, color='#00ff41', fill=False, linestyle='--', alpha=0.5)
    ax_gg.add_artist(circle)
    
    if uploaded_file and 'lat_g' in df.columns and 'accel' in df.columns:
        ax_gg.scatter(df['lat_g'], df['accel'], c='white', alpha=0.6, s=8)
    
    ax_gg.set_xlim(-1.5, 1.5); ax_gg.set_ylim(-1.5, 1.5)
    ax_gg.axhline(0, color='grey', lw=1, alpha=0.3)
    ax_gg.axvline(0, color='grey', lw=1, alpha=0.3)
    ax_gg.set_xlabel("Lateral G (Cornering)")
    ax_gg.set_ylabel("Longitudinal G (Brake/Accel)")
    st.pyplot(fig_gg)

# Metrics Grid
m1, m2, m3, m4 = st.columns(4)
m1.metric("WIN PROB", f"{prob}%")
m2.metric("AIR DENSITY", f"{rho} kg/m³")
m3.metric("PWR/WGT", f"{round(p_to_w, 3)}")
m4.metric("DA ESTIMATE", f"{int((1.225 - rho) * 10000)} ft")

# 3. AI Sector Strategy Brief
st.markdown("---")
if GOOGLE_API_KEY:
    if st.button("GENERATE SECTOR STRATEGY BRIEF"):
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-flash-latest')
        prompt = f"""
        Role: Lead Race Engineer. 
        Track: {track_input}. Environment: {rho} density. 
        Vehicle: {power}HP / {weight}kg.
        Provide two tactical tips for the driver: 
        1. One for low-speed exit traction.
        2. One for high-speed aero stability.
        Keep it professional and concise.
        """
        response = model.generate_content(prompt)
        st.info(f"**Pit Wall Analysis:** {response.text}")
else:
    st.warning("API Key missing in Secrets.")

st.caption(f"Elite-Racing-Agent v2.5 | {track_input} Optimized | Developer: Championship Edition")
