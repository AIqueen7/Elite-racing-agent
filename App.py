import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai
from PIL import Image

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="Elite-Racing-Agent PRO", page_icon="🏁", layout="wide")

# Enhanced Pro-Racing CSS
st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    [data-testid="stMetricValue"] { font-size: 42px !important; font-family: 'Courier New'; }
    .status-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; }
    .stButton>button { height: 3em; background-color: #00ff41; color: black; }
    </style>
    """, unsafe_allow_html=True)

# --- API & SESSION STATE ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🕹️ COMMAND CENTER")
    driver_mode = st.toggle("COCKPIT MODE (High Vis)", value=False)
    
    st.header("Car Setup")
    weight = st.number_input("Mass (kg)", value=850)
    power = st.number_input("Power (HP)", value=600)
    cd = st.slider("Aero (Cd)", 0.2, 0.9, 0.45)
    
    st.header("Track: Strawberry Creek")
    if st.button("SYNC LIVE DA"):
        url = f"http://api.openweathermap.org/data/2.5/weather?lat=53.3377&lon=-114.1603&appid={WEATHER_API_KEY}&units=metric"
        try:
            res = requests.get(url).json()
            temp_k = res['main']['temp'] + 273.15
            press_pa = res['main']['pressure'] * 100
            st.session_state['rho'] = round(press_pa / (287.05 * temp_k), 4)
            st.success("Density Synced")
        except: st.error("Weather Offline")

# --- DATA PROCESSING ---
rho = st.session_state['rho']
p_to_w = power / weight
perf_ratio = (p_to_w * (rho / 1.225)) / 0.75
prob = min(99, max(10, int(perf_ratio * 100)))

# Pit-Wall Status Logic
if prob > 90: status, color, msg = "🟢 SEND IT", "#00ff41", "Car potential at 95%+. Track conditions optimal."
elif prob > 75: status, color, msg = "🟡 CAUTION", "#ffff00", "Aero drag increasing. Monitor tire temps."
else: status, color, msg = "🔴 BOX BOX", "#ff0000", "Power loss or High DA detected. Check mechanicals."

# --- MAIN UI ---
if not driver_mode:
    st.title("🏁 ELITE-RACING-AGENT | PRO-SERIES")
else:
    st.title("⏱️ LIVE TELEMETRY")

# Status Bar
st.markdown(f'<div class="status-box" style="background-color: {color}; color: black;">{status} | {msg}</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # 1. Performance Map (The Digital Twin)
    v_kmh = np.linspace(10, 250, 100)
    a_g = ( (power*745.7)/(v_kmh/3.6) - 0.5*rho*(v_kmh/3.6)**2*cd*1.5 ) / (weight*9.81)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('dark_background')
    ax.plot(v_kmh, a_g, color='#00ff41', alpha=0.8, linewidth=4, label="LIMIT")
    
    uploaded_file = st.file_uploader("Drop Telemetry CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'speed' in df.columns and 'accel' in df.columns:
            # Color code points by efficiency
            ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=15, label="ACTUAL")
    
    ax.set_ylabel("G-Force")
    ax.set_xlabel("Speed km/h")
    ax.grid(True, alpha=0.1)
    st.pyplot(fig)

with col2:
    # 2. Friction Circle (G-G Diagram)
    st.subheader("G-G Friction Circle")
    fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
    circle = plt.Circle((0, 0), 1.2, color='#00ff41', fill=False, linestyle='--')
    ax_gg.add_artist(circle)
    
    if uploaded_file and 'lat_g' in df.columns:
        ax_gg.scatter(df['lat_g'], df['accel'], c='white', alpha=0.5, s=5)
    
    ax_gg.set_xlim(-1.5, 1.5); ax_gg.set_ylim(-1.5, 1.5)
    ax_gg.axhline(0, color='grey', lw=1); ax_gg.axvline(0, color='grey', lw=1)
    ax_gg.set_xlabel("Lat G (Cornering)"); ax_gg.set_ylabel("Lon G (Accel/Brake)")
    st.pyplot(fig_gg)

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("WIN PROB", f"{prob}%")
m2.metric("AIR DENSITY", f"{rho}")
m3.metric("PWR/WGT", f"{round(p_to_w, 2)}")
m4.metric("DENSITY ALT", "2,150 ft")

# 3. AI Sector Strategy
if GOOGLE_API_KEY and st.button("GET SECTOR STRATEGY"):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
    prompt = f"Race Engineer: {power}HP at {rho} density. Track: Strawberry Creek. Tips for Sector 1 (Launch) and Sector 2 (High Speed Sweeper)."
    st.info(model.generate_content(prompt).text)
