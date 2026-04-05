import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. PRO-SPEC ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | Precision Spec", page_icon="🏁", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #d1d1d1; }
    [data-testid="stMetricValue"] { font-size: 34px !important; color: #00e5ff; font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #111; color: #666; border: 1px solid #222; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #ffffff; border-bottom: 2px solid #00e5ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GLOBAL TRACK DATABASE ---
TRACKS = {
    "Strawberry Creek Raceway (Home)": {"lat": 53.3377, "lon": -114.1603, "alt": 2300},
    "Pikes Peak - Start": {"lat": 38.8405, "lon": -104.9442, "alt": 9390},
    "Pikes Peak - Summit": {"lat": 38.8405, "lon": -105.0445, "alt": 14115},
    "Nürburgring Nordschleife": {"lat": 50.3341, "lon": 6.9427, "alt": 2000},
    "Laguna Seca": {"lat": 36.5841, "lon": -121.7533, "alt": 800},
    "Mount Panorama (Bathurst)": {"lat": -33.4475, "lon": 149.559, "alt": 2800}
}

# --- 3. PHYSICS ENGINE ---
def sim_physics(power, weight, rho, cd, mu, v_range):
    v_ms = v_range / 3.6
    eff_hp = power * ((rho / 1.225) ** 0.7)
    p_w = eff_hp * 745.7
    gs = []
    for v in v_ms:
        v = max(v, 1.0)
        drag = 0.5 * rho * (v**2) * cd * 1.5
        rolling = weight * 9.81 * 0.015
        net_f = ((p_w / v) * 0.85) - drag - rolling
        gs.append(max(min(net_f / (weight * 9.81), mu), -mu))
    return gs, eff_hp

# --- 4. SIDEBAR & DATA INPUT ---
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🛠️ CHASSIS LAB")
    venue_key = st.selectbox("Track Selection", list(TRACKS.keys()))
    v_data = TRACKS[venue_key]
    
    with st.expander("Mechanical Parameters", expanded=True):
        hp = st.number_input("Base BHP", 100, 2500, 600)
        kg = st.number_input("Mass (kg)", 500, 3000, 850)
        mu = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
        cd = st.slider("Aero Drag (Cd)", 0.1, 1.5, 0.45)

    if st.button("SYNC REMOTE ATMOSPHERE"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={v_data['lat']}&lon={v_data['lon']}&appid={st.secrets.get('OPENWEATHER_API_KEY')}&units=metric"
            res = requests.get(url).json()
            tk = res['main']['temp'] + 273.15
            st.session_state['rho'] = round((res['main']['pressure']*100)/(287.05*tk), 4)
            st.success("Atmosphere Locked")
        except: st.error("Sensor Sync Failed")

# --- 5. DATA PRE-PROCESSING ---
rho = st.session_state['rho']
v_ref = np.linspace(5, 320, 100)
curve, cur_hp = sim_physics(hp, kg, rho, cd, mu, v_ref)

f = st.file_uploader("📂 Upload Telemetry CSV (Master Dataset)", type="csv")
df = None
if f:
    df = pd.read_csv(f)
    if 'lat_g' in df.columns and 'accel' in df.columns:
        df['g_sum'] = np.sqrt(df['lat_g']**2 + df['accel']**2)

# --- 6. THE MASTER DASHBOARD ---
t1, t2, t3 = st.tabs(["📊 PERFORMANCE SUMMARY", "🧬 PHYSICS & SENSITIVITY", "🤖 CHIEF AGENT"])

with t1:
    # ROW 1: THE BIG FIVE (Key Metrics)
    c1, c2, c3, c4, c5 = st.columns(5)
    
    da = int((1.225 - rho) * 10000)
    c1.metric("DENSITY ALTITUDE", f"{da} ft")
    
    v_aero = int(np.sqrt((kg * 9.81) / (0.5 * rho * (cd * 2.5) * 1.5)) * 3.6)
    c2.metric("AERO CROSSOVER", f"{v_aero} kmh", "1.0G Load")
    
    c3.metric("EFFECTIVE BHP", f"{int(cur_hp)} hp", f"{int(cur_hp - hp)} loss")
    
    vmax = int(np.cbrt((cur_hp * 745.7 * 0.85) / (0.5 * rho * cd * 1.5)) * 3.6)
    c4.metric("REAL V-MAX", f"{vmax} kmh", f"{int(vmax*0.621)} mph")

    if df is not None and 'g_sum' in df.columns:
        util = (df['g_sum'].max() / mu) * 100
        c5.metric("GRIP UTILIZATION", f"{round(util, 1)}%", "Peak vs. Limit")
    else:
        c5.metric("GRIP UTILIZATION", "N/A")

    # ROW 2: PRIMARY GRAPHS
    mc, sc = st.columns([2, 1])
    with mc:
        st.subheader("Performance Envelope Analysis")
        fig, ax = plt.subplots(figsize=(10, 4.5)); plt.style.use('dark_background')
        ax.plot(v_ref, curve, color='#00e5ff', lw=2.5, label="Physics Twin")
        if df is not None and 'speed' in df.columns and 'accel' in df.columns:
            ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=10, alpha=0.4, label="Session")
