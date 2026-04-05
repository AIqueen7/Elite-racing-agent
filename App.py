import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. PRO-SPEC ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | Master Builder", page_icon="🏁", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #080808; color: #d1d1d1; }
    [data-testid="stMetricValue"] { font-size: 38px !important; color: #00e5ff; font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; background-color: #111; border-radius: 2px; color: #666; border: 1px solid #222;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #ffffff; border-bottom: 2px solid #00e5ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINEERING CORE ---
def simulate_load(power, weight, rho, cd, mu, v_range):
    v_ms = v_range / 3.6
    p_watts = power * 745.7
    accel_gs = []
    for v in v_ms:
        drag = 0.5 * rho * (v**2) * cd * 1.5
        force = (p_watts / v) if v > 0 else 0
        net_g = (force - drag) / (weight * 9.81)
        accel_gs.append(min(net_g, mu))
    return accel_gs

# --- 3. INPUTS ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("⚙️ CHASSIS CONFIG")
    track_id = st.text_input("Venue", "Strawberry Creek Raceway")
    
    with st.expander("Mechanical Specs", expanded=True):
        power = st.number_input("Brake Horsepower (BHP)", 100, 2500, 600)
        mass = st.number_input("Race Mass (kg)", 500, 3000, 850)
        mu_static = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.2)
        drag_coeff = st.slider("Aero Efficiency (Cd)", 0.15, 0.95, 0.42)

    if st.button("SYNC TRACK ATMOSPHERE"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat=53.3377&lon=-114.1603&appid={WEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            temp_k = res['main']['temp'] + 273.15
            st.session_state['rho'] = round((res['main']['pressure']*100) / (287.05 * temp_k), 4)
        except: st.error("Sensor Sync Offline")

rho = st.session_state['rho']

# --- 4. THE INTERFACE ---
tab_live, tab_aero, tab_engineer = st.tabs(["📊 SYSTEM TELEMETRY", "🧬 AERO & KINETICS", "🤖 CHIEF ENGINEER"])

with tab_live:
    col_a, col_b, col_c, col_d = st.columns(4)
    # A veteran wants to see "Density Altitude" and "Power-to-Weight"
    da = int((1.225 - rho) * 10000)
    col_a.metric("DENSITY ALTITUDE", f"{da} ft")
    col_b.metric("AIR DENSITY (ρ)", f"{rho}")
    col_c.metric("WGT/PWR RATIO", f"{round(mass/power, 2)} kg/hp")
    col_d.metric("EST. TOP SPEED", f"{int(np.sqrt((power*745.7)/(0.5*rho*drag_coeff*1.5)) * 3.6)} km/h")

    main_c, side_c = st.columns([2, 1])
    
    with main_c:
        st.subheader("Performance Envelope: Digital Twin vs. Driver")
        v_ref = np.linspace(5, 280, 100)
        twin_data = simulate_load(power, mass, rho, drag_coeff, mu_static, v_ref)
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plt.style.use('dark_background')
        ax.plot(v_ref, twin_data, color='#00e5ff', linewidth=2, label="Theoretical Limit", alpha=0.8)
        
        file = st.file_uploader("Drop Telemetry Data (CSV)", type="csv")
        if file:
            df = pd.read_csv(file)
            if 'speed' in df.columns and 'accel' in df.columns:
                sc = ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=8, alpha=0.4, label="Actual Run")
        ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("Longitudinal G"); ax.legend(); st.pyplot(fig)

    with side_c:
        st.subheader("G-G Friction Circle")
        fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
        # Draw the Limit Circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax_gg.plot(mu_static*np.cos(theta), mu_static*np.sin(theta), color='#00e5ff', linestyle='--', alpha=0.6, label="Tire Limit")
        
        if file:
            # THIS IS THE KEY: Plotting actual lateral vs longitudinal data
            if 'lat_g' in df.columns and 'accel' in df.columns:
                ax_gg.scatter(df['lat_g'], df['accel'], color='white', s=2, alpha=0.3, label="Driver Path")
        
        ax_gg.set_xlim(-mu_static-0.2, mu_static+0.2); ax_gg.set_ylim(-mu_static-0.2, mu_static+0.2)
        ax_gg.set_xlabel("Lateral G"); ax_gg.set_ylabel("Long. G")
        st.pyplot(fig_gg)

with tab_aero:
    st.header("Aero Sensitivity & Load Mapping")
    v_map = np.linspace(0, 300, 100)
    # Calculate Downforce (Simplified for a builder)
    downforce = [0.5 * rho * (v/3.6)**2 * (drag_coeff * 2.5) * 1.5 for v in v_map]
    
    fig_aero, ax_aero = plt.subplots(figsize=(10, 4))
    ax_aero.fill_between(v_map, downforce, color='#00e5ff', alpha=0.15, label="Total Downforce (N)")
    ax_aero.set_title("Vertical Load vs. Velocity")
    ax_aero.set_xlabel("Speed (km/h)"); ax_aero.set_ylabel("Newtons"); ax_aero.legend(); st.pyplot(fig_aero)
    
    st.markdown("""
    ### Chassis Insights
    * **Mechanical vs. Aero:** At low speeds, grip is defined by tire compound ($\mu$). At high speeds, the "Aero Map" takes over. 
    * **Load Sensitivity:** Note how the vertical load increases exponentially. This requires stiffer spring rates to prevent the car from bottoming out at top speed.
    """)

with tab_engineer:
    st.subheader("Race Engineering Terminal")
    if query := st.chat_input("Technical inquiry..."):
        with st.chat_message("assistant"):
            if GOOGLE_API_KEY:
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                p_text = f"Consultant for a 40-year veteran car builder. Car: {power}HP, {mass}kg. Focus on physics. Query: {query}"
                st.write(model.generate_content(p_text).text)
            else: st.error("AI Key Missing")

st.caption(f"Elite-Racing-Agent | High-Performance Engineering Console | {track_id}")
