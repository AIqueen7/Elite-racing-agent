import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai
from datetime import datetime

# --- 1. SYSTEM ARCHITECTURE & ENGINE CONFIG ---
st.set_page_config(page_title="Elite-Racing-Agent | Architect Spec", page_icon="🏎️", layout="wide")

# High-Contrast "Night Vision" UI for Pits/Cockpit
st.markdown("""
    <style>
    .main { background-color: #020202; color: #e0e0e0; }
    [data-testid="stMetricValue"] { font-size: 32px !important; color: #00e5ff; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #0a0a0a; border: 1px solid #1a1a1a; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #00e5ff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GLOBAL TRACK & ELEVATION DATABASE ---
TRACKS = {
    "Strawberry Creek Raceway (Home)": {"lat": 53.3377, "lon": -114.1603, "alt": 2300},
    "Pikes Peak - Start": {"lat": 38.8405, "lon": -104.9442, "alt": 9390},
    "Pikes Peak - Summit": {"lat": 38.8405, "lon": -105.0445, "alt": 14115},
    "Nürburgring Nordschleife": {"lat": 50.3341, "lon": 6.9427, "alt": 2000}
}

# --- 3. CORE PHYSICS KERNEL (DIGITAL TWIN) ---
def twin_kernel(hp, mass, rho, cd, mu, v_range):
    v_ms = v_range / 3.6
    # 0.7 exponent accounts for NA vs Forced Induction altitude sensitivity
    eff_hp = hp * ((rho / 1.225) ** 0.7)
    p_w = eff_hp * 745.7
    gs = []
    for v in v_ms:
        v = max(v, 1.0)
        drag = 0.5 * rho * (v**2) * cd * 1.5 # 1.5 = Reference Area (m2)
        net_f = ((p_w / v) * 0.88) - drag - (mass * 9.81 * 0.012) # 0.88 = Drivetrain Efficiency
        gs.append(max(min(net_f / (mass * 9.81), mu), -mu))
    return gs, eff_hp

# --- 4. DATA PIPELINE & INPUTS ---
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🎛️ SYSTEM COMMAND")
    venue = st.selectbox("Active Mission Environment", list(TRACKS.keys()))
    v_data = TRACKS[venue]
    
    with st.expander("Chassis & Powertrain DNA", expanded=True):
        hp_base = st.number_input("Nominal BHP", 100, 2500, 600)
        kg_total = st.number_input("Mission Mass (kg)", 500, 3000, 850)
        mu_static = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
        cd_aero = st.slider("Drag Coefficient (Cd)", 0.1, 1.5, 0.45)

    if st.button("SYNC ENVIRONMENTAL TELEMETRY"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={v_data['lat']}&lon={v_data['lon']}&appid={st.secrets.get('OPENWEATHER_API_KEY')}&units=metric"
            res = requests.get(url).json()
            tk = res['main']['temp'] + 273.15
            st.session_state['rho'] = round((res['main']['pressure']*100)/(287.05*tk), 4)
            st.success("Atmospheric Synchronization Complete")
        except: st.error("IoT Gateway Offline: Check API Key")

# --- 5. REAL-TIME PROCESSING ---
rho = st.session_state['rho']
v_ref = np.linspace(5, 340, 120)
physics_curve, effective_bhp = twin_kernel(hp_base, kg_total, rho, cd_aero, mu_static, v_ref)

# Telemetry Ingestion
f = st.file_uploader("📥 Synchronize Telemetry Stream (CSV/IoT)", type="csv")
df = None
if f:
    df = pd.read_csv(f)
    if 'lat_g' in df.columns and 'accel' in df.columns:
        df['g_sum'] = np.sqrt(df['lat_g']**2 + df['accel']**2)

# --- 6. ARCHITECTURAL INTERFACE ---
tabs = st.tabs(["🏛️ SYSTEM ARCHITECTURE", "📊 LIVE TELEMETRY", "🧬 PREDICTIVE DYNAMICS", "🤖 CHIEF AGENT"])

# NEW: Architectural Overview Tab
with tabs[0]:
    st.header("Digital Twin Architecture: Modular Edge/Cloud Hybrid")
    c_arch1, c_arch2 = st.columns(2)
    with c_arch1:
        st.subheader("Model Selection & Justification")
        st.markdown("""
        * **Vehicle Dynamics:** 6-DOF Multi-Body Physics Engine (Real-Time).
        * **Aerodynamics:** Reinforcement Learning (RL) for dynamic wing-angle optimization.
        * **Maintenance:** Time-Series Forecasting (LSTM) for thermal fatigue prediction.
        * **Aero-BHP Logic:** $BHP_{Eff} = BHP_{Base} \cdot (\rho / \rho_0)^{0.7}$.
        """)
    with c_arch2:
        st.subheader("Data Pipeline Design")
        st.markdown("""
        1.  **Edge Layer:** On-car sensors (CAN-bus) push to local cache.
        2.  **Transport:** 5G/Telemetry Link to Cloud Ingestion Service.
        3.  **Inference:** Digital Twin replicates state and predicts "Next-Lap" thermal decay.
        4.  **HMI:** This dashboard serves as the Expert Decision-Support Interface.
        """)

with tabs[1]:
    # Row 1: Key Performance Indicators
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    da = int((1.225 - rho) * 10000)
    kpi1.metric("DENSITY ALTITUDE", f"{da} ft")
    v_crossover = int(np.sqrt((kg_total * 9.81) / (0.5 * rho * (cd_aero * 2.5) * 1.5)) * 3.6)
    kpi2.metric("AERO CROSSOVER", f"{v_crossover} kmh")
    kpi3.metric("EFFECTIVE BHP", f"{int(effective_bhp)} hp")
    vmax = int(np.cbrt((effective_bhp * 745.7 * 0.85) / (0.5 * rho * cd_aero * 1.5)) * 3.6)
    kpi4.metric("REAL V-MAX", f"{vmax} kmh")
    if df is not None:
        util = (df['g_sum'].max() / mu_static) * 100
        kpi5.metric("GRIP UTILIZATION", f"{round(util, 1)}%")
    else: kpi5.metric("GRIP UTILIZATION", "N/A")

    # Primary Visualization
    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.subheader("Digital Twin vs. Physical Reality")
        fig, ax = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
        ax.plot(v_ref, physics_curve, color='#00e5ff', lw=2, label="Twin Prediction")
        if df is not None:
            ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=8, alpha=0.3, label="IoT Sensor Stream")
        ax.set_xlabel("Velocity (km/h)"); ax.set_ylabel("Longitudinal G"); ax.legend(); st.pyplot(fig)

with tabs[2]:
    st.header("Predictive Scenarios & Maintenance")
    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Aero Efficiency Vector")
        aero_force = [0.5 * rho * (v/3.6)**2 * (cd_aero * 2.5) for v in v_ref]
        fig_a, ax_a = plt.subplots(); plt.style.use('dark_background')
        ax_a.plot(v_ref, aero_force, color='#00e5ff', label="Downforce (N)")
        ax_a.set_ylabel("Force (Newtons)"); ax_a.set_xlabel("Speed"); st.pyplot(fig_a)
    with p2:
        st.subheader("Predictive Maintenance: Brake Fatigue")
        if df is not None:
            fatigue = np.cumsum(df['g_sum'] * 0.05) # Simulated wear accumulation
            fig_f, ax_f = plt.subplots(); plt.style.use('dark_background')
            ax_f.plot(df.index, fatigue, color='#ff4b4b', label="Component Wear Index")
            ax_f.axhline(80, color='yellow', ls='--', label="Service Warning")
            ax_f.set_ylabel("Wear Index"); ax_f.legend(); st.pyplot(fig_f)

with tabs[3]:
    st.header("🤖 Chief Engineering Agent")
    # Human-AI Collaboration: Subjective Input Layer
    st.subheader("Experiential Input")
    subjective_feel = st.text_input("Jay, enter subjective feedback (e.g., 'Vibration at 180kmh', 'Understeer in Turn 3')")
    
    agent_ctx = f"""
    ROLE: Lead Motorsport Engineer for Jay Esterer.
    STATE: {effective_bhp}HP, {venue}, {rho} air density.
    MODEL OUTPUT: Aero crossover at {v_crossover}kmh. 
    DRIVER FEEDBACK: {subjective_feel}
    OBJECTIVE: Analyze subjective 'feel' alongside objective telemetry to suggest mechanical optimization.
    """
    
    if q := st.chat_input("Request setup optimization or scenario simulation..."):
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                m = genai.GenerativeModel('gemini-1.5-flash')
                st.markdown(m.generate_content(f"{agent_ctx}\n\nQUERY: {q}").text)

st.caption(f"Elite-Racing-Agent Architect Spec | Mission: {venue}")
