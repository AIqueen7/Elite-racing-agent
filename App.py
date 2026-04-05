import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. PRO-SPEC ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | Summit Spec", page_icon="⛰️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #070707; color: #d1d1d1; }
    [data-testid="stMetricValue"] { font-size: 38px !important; color: #00e5ff; font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; background-color: #111; border-radius: 2px; color: #666; border: 1px solid #222;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #ffffff; border-bottom: 2px solid #00e5ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE TRACK DATABASE ---
# Coordinates and Base Altitudes for precise atmospheric modeling
TRACK_DB = {
    "Strawberry Creek Raceway (Home)": {"lat": 53.3377, "lon": -114.1603, "base_alt": 2300},
    "Pikes Peak - Start Line": {"lat": 38.8405, "lon": -104.9442, "base_alt": 9390},
    "Pikes Peak - Glen Cove": {"lat": 38.8850, "lon": -105.0110, "base_alt": 11440},
    "Pikes Peak - Summit": {"lat": 38.8405, "lon": -105.0445, "base_alt": 14115},
    "Mount Washington Auto Road": {"lat": 44.2705, "lon": -71.3033, "base_alt": 6288},
    "Custom / Other": {"lat": 0, "lon": 0, "base_alt": 0}
}

# --- 3. ADVANCED PHYSICS ENGINE ---
def get_altitude_power_factor(rho):
    return (rho / 1.225) ** 0.7 

def simulate_dynamics(power, weight, rho, cd, mu, v_range):
    v_ms = v_range / 3.6
    effective_hp = power * get_altitude_power_factor(rho)
    p_watts = effective_hp * 745.7
    
    accel_gs = []
    for v in v_ms:
        if v < 1: v = 1 
        drag = 0.5 * rho * (v**2) * cd * 1.5
        rolling_res = weight * 9.81 * 0.015 
        force_avail = (p_watts / v) * 0.85 
        
        net_force = force_avail - drag - rolling_res
        net_g = net_force / (weight * 9.81)
        accel_gs.append(max(min(net_g, mu), -mu))
    return accel_gs, effective_hp

# --- 4. SIDEBAR & LOCATION SELECTION ---
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🏔️ MISSION PARAMETERS")
    
    # THE DROPDOWN
    selected_venue = st.selectbox("Select Mission Location", list(TRACK_DB.keys()))
    venue_data = TRACK_DB[selected_venue]
    
    with st.expander("Chassis & Aero DNA", expanded=True):
        base_power = st.number_input("Sea Level BHP", 100, 2500, 600)
        mass = st.number_input("Race Mass (kg)", 500, 3000, 850)
        mu_static = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
        drag_coeff = st.slider("Aero Drag (Cd)", 0.20, 1.20, 0.45)

    if st.button("SYNC ATMOSPHERICS"):
        if selected_venue != "Custom / Other":
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?lat={venue_data['lat']}&lon={venue_data['lon']}&appid={st.secrets.get('OPENWEATHER_API_KEY')}&units=metric"
                res = requests.get(url).json()
                temp_k = res['main']['temp'] + 273.15
                st.session_state['rho'] = round((res['main']['pressure']*100) / (287.05 * temp_k), 4)
                st.success(f"Atmospheric sync complete for {selected_venue}")
            except: st.error("Sensor Sync Offline - Check API Key")
        else:
            st.warning("Please input manual data for custom locations.")

rho = st.session_state['rho']
v_ref = np.linspace(5, 300, 100)
digital_twin, current_bhp = simulate_dynamics(base_power, mass, rho, drag_coeff, mu_static, v_ref)

# --- 5. DASHBOARD ---
tab_telemetry, tab_pikes_peak, tab_engineer = st.tabs(["📊 LIVE TELEMETRY", "🧬 ALTITUDE DYNAMICS", "🤖 CHIEF ENGINEER"])

with tab_telemetry:
    c1, c2, c3, c4 = st.columns(4)
    da = int((1.225 - rho) * 10000)
    c1.metric("DENSITY ALTITUDE", f"{da} ft")
    c2.metric("EFFECTIVE BHP", f"{int(current_bhp)}")
    c3.metric("AIR DENSITY", f"{rho}")
    
    # Real-world top speed (Cubic root accounts for power vs drag exponential)
    top_speed = int(np.cbrt((current_bhp * 745.7 * 0.85) / (0.5 * rho * drag_coeff * 1.5)) * 3.6)
    c4.metric("REALISTIC V-MAX", f"{top_speed} km/h")

    main_c, side_c = st.columns([2, 1])
    
    with main_c:
        st.subheader(f"Performance Envelope: {selected_venue}")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plt.style.use('dark_background')
        ax.plot(v_ref, digital_twin, color='#00e5ff', linewidth=2, label="Theoretical Limit")
        
        file = st.file_uploader("Import Session CSV", type="csv")
        if file:
            df = pd.read_csv(file)
            if 'speed' in df.columns and 'accel' in df.columns:
                ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=8, alpha=0.5, label="Actual Run")
        ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("G-Force"); ax.legend(); st.pyplot(fig)

    with side_c:
        st.subheader("G-G Combined Grip")
        fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
        theta = np.linspace(0, 2*np.pi, 100)
        ax_gg.plot(mu_static*np.cos(theta), mu_static*np.sin(theta), color='#00e5ff', linestyle='--', alpha=0.5)
        
        if file and 'lat_g' in df.columns and 'acc
