import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. SYSTEM ARCHITECTURE & UI ---
st.set_page_config(page_title="Elite-Racing-Agent | Digital Twin", page_icon="🏎️", layout="wide")

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

# --- 3. PHYSICS & TWIN KERNEL ---
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

# --- 4. SIDEBAR CONFIGURATION ---
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🛠️ CHASSIS LAB")
    venue_key = st.selectbox("Track Selection", list(TRACKS.keys()))
    v_data = TRACKS[venue_key]
    
    with st.expander("Mechanical Parameters", expanded=True):
        hp = st.number_input("Base BHP (Sea Level)", 100, 2500, 600)
        kg = st.number_input("Race Mass (kg)", 500, 3000, 850)
        mu = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
        cd = st.slider("Aero Drag (Cd)", 0.1, 1.5, 0.45)

    if st.button("SYNC ATMOSPHERIC TWIN"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={v_data['lat']}&lon={v_data['lon']}&appid={st.secrets.get('OPENWEATHER_API_KEY')}&units=metric"
            res = requests.get(url).json()
            tk = res['main']['temp'] + 273.15
            st.session_state['rho'] = round((res['main']['pressure']*100)/(287.05*tk), 4)
            st.success("Atmosphere Synchronized")
        except: st.error("Sensor Sync Failed")

# --- 5. DATA PIPELINE ---
rho = st.session_state['rho']
v_ref = np.linspace(5, 320, 100)
curve, cur_hp = sim_physics(hp, kg, rho, cd, mu, v_ref)

f = st.file_uploader("📂 Synchronize Telemetry Stream (CSV)", type="csv")
df = None
if f:
    df = pd.read_csv(f)
    if 'lat_g' in df.columns and 'accel' in df.columns:
        df['g_sum'] = np.sqrt(df['lat_g']**2 + df['accel']**2)

# --- 6. THE MULTI-MODAL DASHBOARD ---
t1, t2, t3 = st.tabs(["📊 PERFORMANCE SUMMARY", "🧬 PHYSICS & SENSITIVITY", "🤖 DIGITAL TWIN AGENT"])

with t1:
    c1, c2, c3, c4, c5 = st.columns(5)
    da = int((1.225 - rho) * 10000)
    c1.metric("DENSITY ALTITUDE", f"{da} ft")
    v_aero = int(np.sqrt((kg * 9.81) / (0.5 * rho * (cd * 2.5) * 1.5)) * 3.6)
    c2.metric("AERO CROSSOVER", f"{v_aero} kmh", "1.0G Load")
    c3.metric("EFFECTIVE BHP", f"{int(cur_hp)} hp", f"{int(cur_hp - hp)} loss")
    vmax = int(np.cbrt((cur_hp * 745.7 * 0.85) / (0.5 * rho * cd * 1.5)) * 3.6)
    c4.metric("REAL V-MAX", f"{vmax} kmh")
    
    if df is not None:
        util = (df['g_sum'].max() / mu) * 100
        c5.metric("GRIP UTILIZATION", f"{round(util, 1)}%")
    else: c5.metric("GRIP UTILIZATION", "N/A")

    mc, sc = st.columns([2, 1])
    with mc:
        st.subheader("System Performance Envelope")
        fig, ax = plt.subplots(figsize=(10, 4.5)); plt.style.use('dark_background')
        ax.plot(v_ref, curve, color='#00e5ff', lw=2.5, label="Digital Twin Prediction")
        if df is not None:
            ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=10, alpha=0.4, label="Physical Sensor Data")
        ax.set_xlabel("Velocity (km/h)"); ax.set_ylabel("G-Force"); ax.legend(); st.pyplot(fig)

    with sc:
        st.subheader("G-G Friction Circle")
        fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
        t = np.linspace(0, 2*np.pi, 100)
        ax_gg.plot(mu*np.cos(t), mu*np.sin(t), color='#00e5ff', ls='--', alpha=0.4)
        if df is not None:
            ax_gg.scatter(df['lat_g'], df['accel'], color='white', s=3, alpha=0.3)
        ax_gg.set_xlim(-mu-0.2, mu+0.2); ax_gg.set_ylim(-mu-0.2, mu+0.2); st.pyplot(fig_gg)

with t2:
    st.header("Predictive Sensitivity Analysis")
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Aero-Mechanical Grip Correlation")
        aero_comp = [0.5 * rho * (v/3.6)**2 * (cd * 2.0) / (kg * 9.81) for v in v_ref]
        fig1, ax1 = plt.subplots(); plt.style.use('dark_background')
        ax1.fill_between(v_ref, mu, color='#222', label="Mechanical Base")
        ax1.plot(v_ref, aero_comp, color='#00e5ff', lw=2, label="Aero Vector")
        ax1.set_xlabel("Speed"); ax1.set_ylabel("G-Potential"); ax1.legend(); st.pyplot(fig1)
    with g2:
        st.subheader("Thermal Energy Flux")
        if df is not None:
            energy = df['g_sum'] * df['speed']
            fig2, ax2 = plt.subplots(); plt.style.use('dark_background')
            ax2.plot(df.index, energy, color='#ff4b4b')
            ax2.set_ylabel("Work (Force x Velocity)"); st.pyplot(fig2)

with t3:
    st.header("🤖 High-Fidelity AI Inference Engine")
    st.info("Agent Role: Systems Architect & Motorsport Engineer for Jay Esterer")
    
    # AI System Prompt Logic
    system_instruction = f"""
    You are an elite AI Systems Architect and Motorsport Engineer. 
    Context: You are managing a Digital Twin for Jay Esterer (40+ years experience).
    Vehicle Stats: {hp} BHP (Base), {kg}kg mass, {cd} Drag Coefficient.
    Track Context: {venue_key} at {v_data['alt']}ft elevation.
    Current Physics: {cur_hp} Effective BHP, {v_aero}km/h Aero Crossover.
    
    Task: Provide engineering-grade insights. Avoid oversimplification. 
    Analyze predictive states, setup optimizations, and driver behavior modeling.
    """
    
    if q := st.chat_input("Input feedback, query system state, or request setup optimization..."):
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                m = genai.GenerativeModel('gemini-1.5-flash')
                response = m.generate_content(f"{system_instruction}\n\nUser Input: {q}")
                st.markdown(response.text)
            else: st.error("Inference Engine Offline: Check API Configuration")

st.caption(f"Digital Twin Environment | Precision Engineering | Track: {venue_key}")
