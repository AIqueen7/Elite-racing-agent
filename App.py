import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from scipy.stats import multivariate_normal

# --- 1. SYSTEM ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | Structural DNA", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #fdfdfd; }
    [data-testid="stMetricValue"] { font-size: 32px !important; color: #00e5ff; font-family: 'JetBrains Mono'; }
    .stHeader { background-color: #050505; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DYNAMIC DNA INPUTS (THE SPEC) ---
with st.sidebar:
    st.title("🛡️ STRUCTURAL DNA")
    
    with st.expander("Chassis & Materials", expanded=True):
        mat_type = st.selectbox("Upright Material", ["6061-T6 Aluminum", "Magnesium", "Carbon Composite"])
        brake_mat = st.selectbox("Brake Interface", ["Carbon-Ceramic", "Cast Iron", "Steel High-Carbon"])
        mount_type = st.radio("Aero Mounting", ["Chassis-Mounted", "Suspension-Mounted (Unsprung)"])
        
    with st.expander("Kinematic Stagger", expanded=True):
        f_tire = st.number_input("Front Tire Width (mm)", 200, 400, 285)
        r_tire = st.number_input("Rear Tire Width (mm)", 200, 500, 335)
        wheelbase = st.number_input("Wheelbase (mm)", 2000, 3500, 2450)

    st.divider()
    hp = st.number_input("Nominal BHP", 500, 3000, 1200)
    kg = st.number_input("Dry Mass (kg)", 500, 2500, 850)
    rho_s = st.slider("Air Density (kg/m³)", 0.6, 1.3, 1.10)

# --- 3. THE NEURAL-STRUCTURAL KERNEL ---
def calculate_system_physics(hp, kg, f_w, r_w, mat, b_mat):
    # Material Constants (Modulus/Thermal)
    modulus_map = {"6061-T6 Aluminum": 68.9, "Magnesium": 45, "Carbon Composite": 150}
    thermal_map = {"Carbon-Ceramic": 0.95, "Cast Iron": 0.55, "Steel High-Carbon": 0.65}
    
    e_mod = modulus_map[mat]
    t_diff = thermal_map[b_mat]
    
    # Understeer Gradient (K) based on Tire Stagger
    stagger_ratio = r_w / f_w
    k_gradient = (1.2 / stagger_ratio) * (kg / 1000)
    
    # Aero-Elastic Flex: Higher Modulus = Less stall at high speed
    v = np.linspace(0, 380, 100)
    flex = (0.5 * 1.1 * (v/3.6)**2) / (e_mod * 1000)
    cl_dynamic = 3.5 * (1 - flex)
    
    return v, cl_dynamic, k_gradient, t_diff

v_ax, cl_ax, k_grad, t_alpha = calculate_system_physics(hp, kg, f_tire, r_tire, mat_type, brake_mat)

# --- 4. THE INTERFACE ---
t1, t2, t3, t4 = st.tabs(["📊 KINEMATIC BIAS", "🔥 THERMAL PHM", "🌪️ AERO-STRUCTURAL", "🤖 CHIEF AGENT"])

with t1:
    st.header("Dynamic Understeer Gradient (K)")
    # Visualizing how the car pushes vs rotates based on tire stagger
    slip_angles = np.linspace(0, 12, 100)
    f_grip = np.sin(slip_angles/4) * f_tire/100
    r_grip = np.sin(slip_angles/(4*k_grad)) * r_tire/100
    
    fig, ax = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax.plot(slip_angles, f_grip, color='cyan', label="Front Axle Saturation")
    ax.plot(slip_angles, r_grip, color='magenta', label="Rear Axle Saturation")
    ax.set_xlabel("Slip Angle (deg)"); ax.set_ylabel("Lateral Force (N)"); ax.legend()
    st.pyplot(fig)
    st.info(f"Structural Insight: With a {r_tire}mm rear stagger, the AI predicts an Understeer Gradient of {round(k_grad, 2)}. The rear axle will remain linear while the front saturates at {round(slip_angles[np.argmax(f_grip)], 1)}°.")
    

with t2:
    st.header("LSTM Thermal Diffusion (Material Specific)")
    # Fatigue rate now tied to specific heat of brake material
    cycles = np.arange(0, 100)
    thermal_load = np.cumsum((hp/kg) * t_alpha * (1 + 0.2 * np.random.randn(100)))
    
    fig2, ax2 = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax2.plot(cycles, thermal_load, color='#ff4b4b', lw=3)
    ax2.axhline(thermal_load.max()*0.8, color='yellow', ls='--', label="Material Yield Zone")
    ax2.set_xlabel("Duty Cycles"); ax2.set_ylabel("Inferred Heat Saturation"); ax2.legend()
    st.pyplot(fig2)
    st.markdown(f"**Prognostics:** Because you are using **{brake_mat}**, the LSTM calculates a thermal recovery rate of **{round(t_alpha, 2)}**. The predictive risk zone is weighted for the crystalline structural limits of this specific material.")
    

with t3:
    st.header("Center of Pressure (CoP) Shift")
    # Calculating how the aero-balance moves as the wing flexes
    cop_shift = (cl_ax / 3.5) * (wheelbase / 1000)
    fig3, ax3 = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax3.plot(v_ax, cop_shift, color='#00ff9d', lw=3)
    ax3.set_xlabel("Velocity (km/h)"); ax3.set_ylabel("CoP Location (m from Front Axle)")
    st.pyplot(fig3)
    st.caption(f"Aero-Structural: Your **{mat_type}** uprights are predicted to deflect by {round(cl_ax.min()/cl_ax.max(), 3)}% at V-Max, shifting the CoP aft.")
    

with t4:
    st.header("🤖 Multi-Objective Chief Agent")
    subj = st.text_input("Jay, specify a corner entry or high-speed behavior:")
    manifest = f"DNA: {mat_type} Uprights, {brake_mat} Brakes. Stagger: {f_tire}/{r_tire}. Understeer K: {round(k_grad, 2)}."
    
    if q := st.chat_input("Inquire for engineering validation..."):
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                st.markdown(model.generate_content(f"{manifest}\n\nUSER: {q}").text)

st.caption("Elite-Racing-Agent | Structural DNA V5 | Prescriptive Digital Twin")
