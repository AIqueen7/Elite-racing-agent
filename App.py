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

# --- 2. ADVANCED PHYSICS ENGINE (SUMMIT SPEC) ---
def get_altitude_power_factor(rho):
    """Calculates power loss. 1.225 is sea level density."""
    return (rho / 1.225) ** 0.7 # Turbocharged engines regain some but still lose efficiency

def simulate_dynamics(power, weight, rho, cd, mu, v_range):
    v_ms = v_range / 3.6
    # Adjusted Power for Altitude
    effective_hp = power * get_altitude_power_factor(rho)
    p_watts = effective_hp * 745.7
    
    accel_gs = []
    for v in v_ms:
        if v < 1: v = 1 # Avoid division by zero
        drag = 0.5 * rho * (v**2) * cd * 1.5
        rolling_res = weight * 9.81 * 0.015 # Friction of tires on tarmac
        force_avail = (p_watts / v) * 0.85 # 15% Drivetrain loss
        
        net_force = force_avail - drag - rolling_res
        net_g = net_force / (weight * 9.81)
        accel_gs.append(max(min(net_g, mu), -mu))
    return accel_gs, effective_hp

# --- 3. INPUTS ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🏔️ SUMMIT CONFIG")
    track_id = st.text_input("Venue", "Strawberry Creek / Pikes Peak")
    
    with st.expander("Mechanical DNA", expanded=True):
        base_power = st.number_input("Sea Level BHP", 100, 2500, 600)
        mass = st.number_input("Race Mass (kg)", 500, 3000, 850)
        mu_static = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
        drag_coeff = st.slider("Aero Drag (Cd)", 0.20, 1.20, 0.45)

    if st.button("SYNC ATMOSPHERICS"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat=53.3377&lon=-114.1603&appid={WEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            temp_k = res['main']['temp'] + 273.15
            st.session_state['rho'] = round((res['main']['pressure']*100) / (287.05 * temp_k), 4)
        except: st.error("Sensor Offline")

rho = st.session_state['rho']
v_ref = np.linspace(5, 300, 100)
digital_twin, current_bhp = simulate_dynamics(base_power, mass, rho, drag_coeff, mu_static, v_ref)

# --- 4. DASHBOARD ---
tab_telemetry, tab_pikes_peak, tab_engineer = st.tabs(["📊 LIVE TELEMETRY", "🧬 ALTITUDE DYNAMICS", "🤖 CHIEF ENGINEER"])

with tab_telemetry:
    c1, c2, c3, c4 = st.columns(4)
    da = int((1.225 - rho) * 10000)
    c1.metric("DENSITY ALTITUDE", f"{da} ft")
    c2.metric("EFFECTIVE BHP", f"{int(current_bhp)}")
    c3.metric("AIR DENSITY", f"{rho}")
    # Realistic Top Speed Calculation (Where Drag == Thrust)
    top_speed = int(np.cbrt((current_bhp * 745.7 * 0.85) / (0.5 * rho * drag_coeff * 1.5)) * 3.6)
    c4.metric("REALISTIC V-MAX", f"{top_speed} km/h")

    main_c, side_c = st.columns([2, 1])
    
    with main_c:
        st.subheader("Performance Envelope: Acceleration potential")
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
        
        if file and 'lat_g' in df.columns and 'accel' in df.columns:
            ax_gg.scatter(df['lat_g'], df['accel'], color='white', s=3, alpha=0.4)
            # G-Sum utilization calculation
            g_sum = np.sqrt(df['lat_g']**2 + df['accel']**2).mean()
            st.write(f"Mean Grip Usage: {round((g_sum/mu_static)*100,1)}%")
            
        ax_gg.set_xlim(-mu_static-0.2, mu_static+0.2); ax_gg.set_ylim(-mu_static-0.2, mu_static+0.2)
        ax_gg.set_xlabel("Lat G"); ax_gg.set_ylabel("Long G")
        st.pyplot(fig_gg)

with tab_pikes_peak:
    st.header("Altitude Sensitivity Analysis")
    # Simulate climb from Sea Level to 14,000ft
    altitudes = np.linspace(0, 14000, 50)
    densities = 1.225 * np.exp(-altitudes / 30000)
    power_curve = [base_power * get_altitude_power_factor(d) for d in densities]
    
    fig_climb, ax_climb = plt.subplots(figsize=(10, 4))
    ax_climb.plot(altitudes, power_curve, color='#ff4b4b', linewidth=3)
    ax_climb.set_title("Power Attrition vs. Altitude (The Hill Climb Tax)")
    ax_climb.set_xlabel("Altitude (ft)"); ax_climb.set_ylabel("Available BHP")
    st.pyplot(fig_climb)
    
    st.info("Turbocharged Unlimited cars typically lose ~1-2% power per 1,000ft, whereas naturally aspirated cars lose ~3-4%.")

with tab_engineer:
    st.subheader("Engineering Query Console")
    if query := st.chat_input("Ask about chassis set-up or aero..."):
        with st.chat_message("assistant"):
            if GOOGLE_API_KEY:
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context = f"Consulting for Jay Esterer, 40-yr driver. Car: {base_power}HP, {mass}kg. Setup: Unlimited Class. Query: {query}"
                st.write(model.generate_content(context).text)
            else: st.error("AI Key Missing")

st.caption("Elite-Racing-Agent v8.0 | Summit Spec | Optimized for Strawberry Creek Performance")
