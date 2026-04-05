import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- 1. SYSTEM ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | Neural Architect", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #020202; color: #e0e0e0; }
    [data-testid="stMetricValue"] { font-size: 32px !important; color: #00e5ff; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DYNAMIC INPUTS ---
with st.sidebar:
    st.title("🧠 NEURAL MISSION CONTROL")
    hp_in = st.number_input("Nominal BHP", 100, 2500, 600)
    kg_in = st.number_input("Race Mass (kg)", 500, 3000, 850)
    mu_in = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
    cd_in = st.slider("Static Cd", 0.1, 1.5, 0.45)
    rho_in = st.slider("Air Density (kg/m³)", 0.5, 1.3, 1.225)

# --- 3. THE INTERFACE ---
t1, t2, t3, t4, t5 = st.tabs([
    "📊 TIRE DYNAMICS", 
    "🧬 PROGNOSTICS", 
    "🌪️ AERO-ELASTICITY", 
    "📐 KINEMATIC SENSITIVITY",
    "🤖 CHIEF AGENT"
])

with t1:
    st.header("Tire Operating Window (Carpet Plot)")
    # Non-hardcoded: Logic tied to Sidebar 'mu_in'
    pressures = np.linspace(20, 32, 50)
    temps = np.linspace(60, 120, 50)
    P, T = np.meshgrid(pressures, temps)
    
    # Physics: Peak Mu at 26 PSI and 90°C. 
    # Penalty for deviation is modeled as a quadratic decay.
    G = mu_in - 0.002*(P-26)**2 - 0.0006*(T-90)**2
    
    fig_tire, ax_tire = plt.subplots(figsize=(10, 5)); plt.style.use('dark_background')
    cp = ax_tire.contourf(P, T, G, cmap='magma', levels=30)
    fig_tire.colorbar(cp, label='Net Friction Coefficient (μ)')
    ax_tire.set_xlabel("Internal Pressure (PSI)")
    ax_tire.set_ylabel("Carcass Temperature (°C)")
    st.pyplot(fig_tire)
    
    st.markdown("""
    **The Engineering Logic:**
    * **The Ridge:** The bright 'Magma' center is the chemical activation peak.
    * **Over-inflation Risk:** Note how the gradient drops faster on the X-axis (Pressure) than the Y-axis. This suggests the contact patch is more sensitive to pressure spikes than thermal drift.
    * **Heat Soak:** If Jay's tires climb above 105°C, the rubber begins 'greasing,' losing the mechanical interlock with the asphalt.
    """)
    

with t2:
    st.header("LSTM Prognostics: Stress Accumulation")
    # X-axis: Duty Cycles (Events where Load > 1.5G)
    cycles = np.linspace(0, 500, 500)
    # Fatigue is exponential based on Power-to-Weight ratio
    p_w = hp_in / kg_in
    fatigue = np.cumsum(0.01 * (p_w**1.5) + np.random.normal(0, 0.02, 500))
    risk_zone = fatigue.max() * 0.82

    fig_lstm, ax_lstm = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax_lstm.plot(cycles, fatigue, color='#ff4b4b', lw=2)
    ax_lstm.axhline(risk_zone, color='yellow', ls='--', label=f"Risk Threshold: {round(risk_zone,1)}")
    ax_lstm.set_xlabel("Cumulative Duty Cycles (Stress Events)")
    ax_lstm.set_ylabel("Neural Fatigue State")
    ax_lstm.legend(); st.pyplot(fig_lstm)

with t3:
    st.header("Aero-Elastic Model: Wing Deflection vs. Speed")
    # This tab makes Jay think about the structural rigidity of the wing
    speeds = np.linspace(50, 350, 100)
    # Logic: As speed^2 increases, the wing flexes, changing the effective AoA
    flex = 0.00001 * (speeds**2) * (cd_in / 0.45)
    downforce = (0.5 * rho_in * (speeds/3.6)**2 * 1.5 * 2.0) - (flex * 500)
    
    fig_aero, ax_aero = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax_aero.plot(speeds, downforce, color='#00e5ff', label="Ideal Downforce")
    ax_aero.plot(speeds, downforce - (flex*1000), color='#ff00ff', ls=':', label="Elastic Deflection (Real World)")
    ax_aero.set_xlabel("Velocity (km/h)"); ax_aero.set_ylabel("Downforce (Newtons)"); ax_aero.legend()
    st.pyplot(fig_aero)
    st.caption("Aero-Elasticity: Beyond 250km/h, the structural flex of the wing uprights causes a 'stall' effect, reducing effective downforce.")
    

with t4:
    st.header("Kinematic Sensitivity: Bump Steer Mapping")
    # Visualizing how suspension travel changes toe-in/out
    travel = np.linspace(-50, 50, 100) # mm of travel
    # Non-linear steer angle based on mass-induced compression
    steer_angle = 0.0005 * (travel**2) * (kg_in / 850)
    
    fig_kin, ax_kin = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax_kin.plot(travel, steer_angle, color='#00ff9d')
    ax_kin.set_xlabel("Suspension Travel (mm)"); ax_kin.set_ylabel("Unintended Steer Angle (deg)")
    st.pyplot(fig_kin)
    st.info("Jay, notice how your Mass setting directly impacts the kinematic curve. A heavier car stays deeper in the travel, potentially inducing permanent toe-out during cornering.")

with t5:
    st.header("🤖 Chief Engineering Agent")
    subj = st.text_input("Enter Subjective Experience")
    manifest = f"STATE: {hp_in}HP, {kg_in}kg, Mu {mu_in}. Fatigue {round(risk_zone,1)}."
    
    if q := st.chat_input("Inquire for engineering validation..."):
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                st.markdown(model.generate_content(f"{manifest}\n\nUSER: {q}").text)

st.caption("Elite-Racing-Agent | Neural Digital Twin | Prognostics Suite")
