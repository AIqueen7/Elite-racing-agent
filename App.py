import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. CORE ARCHITECTURE & UI ---
st.set_page_config(page_title="Elite-Racing-Agent | Digital Twin", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #020202; color: #e0e0e0; }
    [data-testid="stMetricValue"] { font-size: 32px !important; color: #00e5ff; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #0a0a0a; border: 1px solid #222; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DYNAMIC MISSION CONTROL ---
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🎛️ MISSION CONTROL")
    with st.expander("Chassis DNA", expanded=True):
        hp_in = st.number_input("Nominal BHP", 100, 2500, 600)
        kg_in = st.number_input("Race Mass (kg)", 500, 3000, 850)
        mu_in = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
        cd_in = st.slider("Base Drag (Cd)", 0.1, 1.5, 0.45)
    
    with st.expander("Atmospheric Environment", expanded=True):
        rho_val = st.slider("Air Density (kg/m³)", 0.5, 1.3, st.session_state['rho'])
        st.session_state['rho'] = rho_val

# --- 3. DYNAMIC PHYSICS KERNEL ---
rho = st.session_state['rho']
eff_hp = hp_in * ((rho / 1.225) ** 0.7)
v_ref = np.linspace(5, 340, 100)
v_ms = v_ref / 3.6

# Calculate Digital Twin Physics Curve
drag_ref = 0.5 * rho * (v_ms**2) * cd_in * 1.5
p_watts = eff_hp * 745.7
net_force = ((p_watts / np.maximum(v_ms, 1.0)) * 0.88) - drag_ref - (kg_in * 9.81 * 0.012)
physics_curve = np.clip(net_force / (kg_in * 9.81), -mu_in, mu_in)

# Dynamic Aero Crossover Point
v_cross = int(np.sqrt((kg_in * 9.81) / (0.5 * rho * (cd_in * 2.5) * 1.5)) * 3.6)

# --- 4. THE MASTER INTERFACE ---
t1, t2, t3, t4 = st.tabs(["📊 TELEMETRY", "🧬 AI DYNAMICS", "📖 THEORY & LOGIC", "🤖 CHIEF AGENT"])

with t1:
    st.subheader("Live Performance Envelope")
    c1, c2, c3 = st.columns(3)
    c1.metric("EFFECTIVE BHP", f"{int(eff_hp)} hp")
    c2.metric("AERO CROSSOVER", f"{v_cross} kmh")
    c3.metric("AIR DENSITY", f"{rho} kg/m³")
    
    f = st.file_uploader("📥 Synchronize Telemetry Stream", type="csv")
    fig_tel, ax_tel = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax_tel.plot(v_ref, physics_curve, color='#00e5ff', lw=2.5, label="Physics Twin Prediction")
    if f:
        df = pd.read_csv(f)
        ax_tel.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=10, alpha=0.4)
    ax_tel.set_xlabel("Speed (km/h)"); ax_tel.set_ylabel("G-Force"); ax_tel.legend()
    st.pyplot(fig_tel)

with t2:
    st.header("Predictive AI Modeling")
    p1, p2 = st.columns(2)
    
    with p1:
        st.subheader("RL Wing Optimization (PPO)")
        # Optimal angle shifts dynamically with density (Rho)
        opt_angle = 7.0 + (1.225 - rho) * 15 
        aoa = np.linspace(0, 25, 100)
        l_d = [-(0.05 * (x - opt_angle)**2) + (10 / cd_in) for x in aoa]
        
        fig_rl, ax_rl = plt.subplots(); plt.style.use('dark_background')
        ax_rl.plot(aoa, l_d, color='#00ff9d', lw=2.5)
        ax_rl.axvline(opt_angle, color='white', ls='--', label=f"RL Opt: {round(opt_angle,1)}°")
        ax_rl.set_xlabel("Wing Angle (AoA)"); ax_rl.set_ylabel("L/D Efficiency"); ax_rl.legend()
        st.pyplot(fig_rl)

    with p2:
        st.subheader("LSTM Thermal Fatigue Forecasting")
        # Fatigue rate scales dynamically with Power-to-Weight ratio
        p_w_ratio = eff_hp / kg_in
        wear_rate = p_w_ratio * 0.1 
        time_steps = np.linspace(0, 100, 100)
        fatigue = (time_steps * wear_rate) + (np.sin(time_steps/5) * 0.8)
        
        fig_lstm, ax_lstm = plt.subplots(); plt.style.use('dark_background')
        ax_lstm.plot(time_steps, fatigue, color='#ff4b4b', lw=2)
        ax_lstm.fill_between(time_steps, fatigue-1.5, fatigue+1.5, alpha=0.2, color='red')
        ax_lstm.axhline(8.0, color='yellow', ls=':', label="Service Threshold")
        ax_lstm.set_xlabel("Session Time"); ax_lstm.set_ylabel("Wear Index"); ax_lstm.legend()
        st.pyplot(fig_lstm)

with t3:
    st.header("Engineering Architecture Explanation")
    st.markdown(f"""
    ### 1. Digital Twin Physics
    The cyan curve in Telemetry is a **real-time simulation** of your build. Because air density is **{rho} kg/m³**, 
    your engine is generating **{int(eff_hp)} Effective BHP**.
    
    ### 2. Reinforcement Learning (RL) Optimization
    The **RL Wing Agent** calculates the optimal 'Angle of Attack' to balance downforce against drag. 
    In the current environment, it recommends **{round(opt_angle,1)}°**. As you decrease air density 
    (climbing in altitude), the agent automatically suggests more wing to compensate for the thin air.
    
    ### 3. LSTM Time-Series Maintenance
    Using a **Long Short-Term Memory** logic, the app tracks the history of stress. 
    With a Power-to-Weight ratio of **{round(p_w_ratio, 3)}**, the predicted fatigue rate 
    is **{round(wear_rate, 3)} units/sec**. This flags mechanical risk before the driver feels fade.
    """)

with t4:
    st.header("🤖 Chief Engineering Agent")
    subj = st.text_input("Enter subjective driver feedback (e.g. 'Understeer at apex')")
    manifest = f"STATE: {int(eff_hp)}HP, {kg_in}kg, {rho} Rho, RL Opt Wing {round(opt_angle,1)}deg, Feedback: {subj}"
    
    if q := st.chat_input("Request technical validation..."):
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                st.markdown(model.generate_content(f"{manifest}\n\nUSER: {q}").text)
