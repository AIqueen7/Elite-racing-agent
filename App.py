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
    .status-box { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 22px; margin-bottom: 20px; border: 2px solid #333; }
    .stButton>button { height: 3.5em; background-color: #00ff41; color: black; border: none; font-weight: bold; width: 100%; border-radius: 10px; }
    .insight-card { background-color: #111; padding: 20px; border-left: 5px solid #00ff41; border-radius: 5px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- API & SESSION STATE ---
# Ensure these are set in your Streamlit Cloud -> Settings -> Secrets
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

# --- SIDEBAR: COMMAND CENTER ---
with st.sidebar:
    st.title("🕹️ COMMAND CENTER")
    weight = st.number_input("Mass (kg)", value=850)
    power = st.number_input("Power (HP)", value=600)
    cd = st.slider("Aero Drag (Cd)", 0.1, 0.9, 0.45)
    mu = st.slider("Tire Grip (mu)", 0.5, 2.0, 1.2)
    track_input = st.text_input("Track", value="Strawberry Creek Raceway")
    
    if st.button("SYNC LIVE WEATHER"):
        try:
            if "Strawberry Creek" in track_input:
                url = f"http://api.openweathermap.org/data/2.5/weather?lat=53.3377&lon=-114.1603&appid={WEATHER_API_KEY}&units=metric"
            else:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={track_input}&appid={WEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            temp_k = res['main']['temp'] + 273.15
            press_pa = res['main']['pressure'] * 100
            st.session_state['rho'] = round(press_pa / (287.05 * temp_k), 4)
            st.success("Density Synced")
        except: st.error("Weather Sync Failed")

rho = st.session_state['rho']
p_to_w = power / weight
prob = min(99, max(10, int((p_to_w * (rho/1.225) / 0.75) * 100)))

# Status Banner Logic
if prob > 90: status, color, msg = "🟢 FULL PUSH", "#00ff41", "Car potential optimal. Send it."
elif prob > 75: status, color, msg = "🟡 MANAGE LOAD", "#ffff00", "DA increasing. Watch intake temps."
else: status, color, msg = "🔴 ABORT LAP", "#ff4b4b", "Critical Density Altitude. Retire the car."

# --- MAIN UI ---
st.title("🏁 ELITE-RACING-AGENT | PRO-SERIES")
st.markdown(f'<div class="status-box" style="background-color: {color}; color: black;">{status} | {msg}</div>', unsafe_allow_html=True)

# --- ACTION BUTTON SECTION ---
if st.button("🚀 GENERATE AI STRATEGY BRIEF"):
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-flash-latest')
            prompt = f"Lead Race Engineer: {power}HP at {rho} density. Track: {track_input}. Give 2 concise tactical tips for the driver."
            response = model.generate_content(prompt)
            st.markdown(f'<div class="insight-card"><b>Engineer Analysis:</b><br>{response.text}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"AI Error: {e}")
    else:
        st.warning("⚠️ ACTION REQUIRED: Add 'GOOGLE_API_KEY' to your Streamlit Secrets to enable this feature.")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Performance Map: Target vs Actual")
    v_kmh = np.linspace(5, 250, 100)
    v_ms = v_kmh / 3.6
    power_w = power * 745.7
    a_g = [min(((power_w / v) - (0.5 * rho * v**2 * cd * 1.5)) / (weight * 9.81), mu) for v in v_ms]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    ax.plot(v_kmh, a_g, color='#00ff41', linewidth=4, label="DIGITAL TWIN")
    
    uploaded_file = st.file_uploader("Upload Telemetry CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'speed' in df.columns and 'accel' in df.columns:
            ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=15, alpha=0.6, label="ACTUAL")
    
    ax.set_ylim(0, 1.6); ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("Longitudinal G")
    ax.legend(); st.pyplot(fig)

with col2:
    st.subheader("G-G Friction Circle")
    fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
    circle = plt.Circle((0, 0), mu, color='#00ff41', fill=False, linestyle='--', linewidth=2)
    ax_gg.add_artist(circle)
    if uploaded_file and 'lat_g' in df.columns and 'accel' in df.columns:
        ax_gg.scatter(df['lat_g'], df['accel'], color='white', s=8, alpha=0.5)
    ax_gg.set_xlim(-1.6, 1.6); ax_gg.set_ylim(-1.6, 1.6)
    ax_gg.axhline(0, color='grey', lw=1, alpha=0.3); ax_gg.axvline(0, color='grey', lw=1, alpha=0.3)
    st.pyplot(fig_gg)

# Metrics Footer
m1, m2, m3, m4 = st.columns(4)
m1.metric("WIN PROB", f"{prob}%")
m2.metric("AIR DENSITY", f"{rho}")
m3.metric("PWR/WGT", f"{round(p_to_w, 3)}")
m4.metric("DENSITY ALT", f"{int((1.225-rho)*10000)} ft")
