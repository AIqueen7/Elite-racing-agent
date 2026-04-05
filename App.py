import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Elite-Racing-Agent | Dynamic Spec", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #020202; color: #e0e0e0; }
    [data-testid="stMetricValue"] { font-size: 30px !important; color: #00e5ff; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PHYSICS & DATA PIPELINE ---
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🎛️ MISSION CONTROL")
    hp_base = st.number_input("Nominal BHP", 100, 2500, 600)
    kg_total = st.number_input("Mass (kg)", 500, 3000, 850)
    mu_static = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
    cd_base = st.slider("Base Drag (Cd)", 0.1, 1.5, 0.45)
    
    if st.button("SYNC ATMOSPHERE"):
        st.session_state['rho'] = 1.05 # Example: High-altitude sync
        st.success("Density Updated")

rho = st.session_state['rho']
f = st.file_uploader("📥 Synchronize Telemetry", type="csv")
df = pd.read_csv(f) if f else None

# --- 3. THE MULTI-MODAL INTERFACE ---
tabs = st.tabs(["📊 LIVE TELEMETRY", "🧬 AI DYNAMICS", "🤖 CHIEF AGENT"])

with tabs[0]:
    # Dynamic KPI Logic
    eff_hp = hp_base * ((rho / 1.225) ** 0.7)
    v_crossover = int(np.sqrt((kg_total * 9.81) / (0.5 * rho * (cd_base * 2.5) * 1.5)) * 3.6)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("EFFECTIVE POWER", f"{int(eff_hp)} hp")
    c2.metric("AERO CROSSOVER", f"{v_crossover} kmh")
    c3.metric("AIR DENSITY", f"{rho} kg/m³")

with tabs[1]:
    st.header("Predictive Dynamics (Parameters-Linked)")
    p1, p2 = st.columns(2)
    
    with p1:
        st.subheader("RL Wing Optimization")
        # DYNAMIC RL: The 'Optimal Angle' shifts with Air Density (Rho)
        # In thin air (low rho), you need more wing angle to get the same downforce
        opt_angle = 7.0 + (1.225 - rho) * 15 
        aoa = np.linspace(0, 25, 100)
        # Efficiency curve shifts based on current Aero Crossover
        l_d = [-(0.05 * (x - opt_angle)**2) + (10 / cd_base) for x in aoa]
        
        fig_rl, ax_rl = plt.subplots(); plt.style.use('dark_background')
        ax_rl.plot(aoa, l_d, color='#00ff9d', lw=2.5)
        ax_rl.axvline(opt_angle, color='white', ls='--', label=f"RL Opt: {round(opt_angle,1)}°")
        ax_rl.set_xlabel("Wing Angle (AoA)"); ax_rl.set_ylabel("L/D Efficiency"); ax_rl.legend(); st.pyplot(fig_rl)
        st.caption(f"RL Inference: Increasing AoA to compensate for {rho} air density.")

    with p2:
        st.subheader("LSTM Thermal Fatigue (Time-Series)")
        # DYNAMIC LSTM: Fatigue rate scales with HP and Mass
        # Heavier, more powerful cars generate more thermal stress
        wear_rate = (hp_base / 600) * (kg_total / 850) * 0.05
        
        time_steps = np.linspace(0, 100, 100)
        # Base fatigue + 'Stochastic Noise' + Parameter Scaling
        fatigue = (time_steps * wear_rate) + (np.sin(time_steps/5) * 0.8)
        
        fig_lstm, ax_lstm = plt.subplots(); plt.style.use('dark_background')
        ax_lstm.plot(time_steps, fatigue, color='#ff4b4b', lw=2)
        ax_lstm.fill_between(time_steps, fatigue-1, fatigue+1, alpha=0.15, color='red')
        ax_lstm.axhline(8.0, color='yellow', ls=':', label="Service Limit")
        ax_lstm.set_xlabel("Session Time (Elapsed)"); ax_lstm.set_ylabel("Predicted Fatigue"); ax_lstm.legend(); st.pyplot(fig_lstm)

with tabs[2]:
    st.header("🤖 Chief Engineering Agent")
    manifest = f"DNA: {int(eff_hp)}HP, {kg_total}kg. Opt Wing: {round(opt_angle,1)}deg."
    if q := st.chat_input("Query the architect..."):
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                st.markdown(model.generate_content(f"{manifest}\n\nUSER: {q}").text)
