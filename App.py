import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- 1. CORE SYSTEM ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | V6 Structural", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #f0f0f0; }
    [data-testid="stMetricValue"] { font-size: 30px !important; color: #00e5ff; font-family: 'JetBrains Mono'; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE STRUCTURAL DNA INPUTS ---
with st.sidebar:
    st.title("🛡️ ARCHITECTURAL DNA")
    
    with st.expander("Unsprung Mass & Metallurgy", expanded=True):
        mat_upright = st.selectbox("Upright Material", ["6061-T6 Aluminum", "Magnesium", "Carbon Composite"])
        wheel_mat = st.selectbox("Wheel Material", ["Forged Aluminum", "Forged Magnesium", "Carbon Fiber"])
        brake_disk = st.selectbox("Brake Rotor", ["Carbon-Ceramic", "High-Carbon Steel", "Cast Iron"])
        
    with st.expander("Kinematic Tire Stagger", expanded=True):
        f_width = st.number_input("Front Width (mm)", 200, 400, 285)
        f_diam = st.number_input("Front Diameter (mm)", 500, 800, 650)
        r_width = st.number_input("Rear Width (mm)", 200, 500, 335)
        r_diam = st.number_input("Rear Diameter (mm)", 500, 800, 680)

    with st.expander("Aero-Structural Config", expanded=True):
        wing_elements = st.radio("Wing Elements", ["Dual-Element", "Triple-Element"])
        aero_mount = st.radio("Mounting Point", ["Chassis-Mounted", "Suspension-Mounted (Unsprung)"])
        wing_material = st.selectbox("Wing Skin", ["Pre-preg Carbon", "Aluminum Honeycomb"])

# --- 3. THE NEURAL-PHYSICS SYNTHESIS ---
def system_synthesis(hp, kg, f_w, r_w, mat, b_mat, mount, elements):
    # Material Constants: Modulus (GPa) & Thermal Expansion (e-6/K)
    phys_props = {
        "6061-T6 Aluminum": {"E": 68.9, "alpha": 23.1},
        "Magnesium": {"E": 45.0, "alpha": 25.0},
        "Carbon Composite": {"E": 150.0, "alpha": 1.0}
    }
    
    # 1. Kinematic Bias (Understeer Gradient K)
    # K = (Wf / Cf) - (Wr / Cr) | Simplified for Stagger Ratio
    stagger_effect = (r_w / f_w) * 1.1 
    k_gradient = (kg / 1000) * (1.5 / stagger_effect)
    
    # 2. Aero-Elastic CoP Shift
    # If suspension mounted, aero load acts directly on the hubs (unsprung)
    # If chassis mounted, it compresses springs (sprung)
    v = np.linspace(0, 360, 200)
    q = 0.5 * 1.10 * (v/3.6)**2 # Dynamic Pressure
    flex_mod = phys_props[mat]["E"]
    elem_penalty = 1.3 if elements == "Triple-Element" else 1.0
    
    # Predicting wing deflection causing CoP shift
    deflection = (q * elem_penalty) / (flex_mod * 1000)
    cop_shift = 0.05 * (deflection**1.5) # meters of shift
    
    return v, k_gradient, cop_shift, phys_props[mat]["alpha"]

v_ax, k_val, cop_ax, thermal_coeff = system_synthesis(1200, 850, f_width, r_width, mat_upright, brake_disk, aero_mount, wing_elements)

# --- 4. THE INTERFACE ---
tabs = st.tabs(["🛞 KINEMATIC BIAS", "🔥 THERMAL PHM", "🌪️ AERO-ELASTICITY", "🤖 CHIEF AGENT"])

with tabs[0]:
    st.header("Dual-Axle Slip Inference (Kinematic Bias)")
    lateral_g = np.linspace(0, 2.5, 100)
    # Modeling saturation: Front tires reach limit before rear based on stagger
    f_sat = np.tanh(lateral_g / (1.2 * (f_width/300)))
    r_sat = np.tanh(lateral_g / (1.5 * (r_width/300)))
    
    fig, ax = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax.plot(lateral_g, f_sat, color='cyan', label="Front Axle Saturation")
    ax.plot(lateral_g, r_sat, color='magenta', label="Rear Axle Saturation")
    ax.fill_between(lateral_g, f_sat, r_sat, color='red', alpha=0.1, label="Understeer Zone")
    ax.set_xlabel("Lateral G-Load"); ax.set_ylabel("Friction Utilization (%)"); ax.legend()
    st.pyplot(fig)
    st.info(f"The Agent predicts the front axle will saturate at {round(lateral_g[np.argmax(f_sat > 0.95)], 2)} Gs. Understeer Gradient (K): {round(k_val, 3)}.")

with tabs[1]:
    st.header("Material-Specific Thermal Mapping")
    # Predicting Camber Gain via thermal expansion of the uprights
    temp_rise = np.linspace(20, 1000, 100) # Brake temp C
    camber_drift = thermal_coeff * temp_rise * 0.0001 # Simplified expansion logic
    
    fig2, ax2 = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax2.plot(temp_rise, camber_drift, color='#ff4b4b', lw=3)
    ax2.set_xlabel("Brake Surface Temp (°C)"); ax2.set_ylabel("Inferred Camber Drift (deg)")
    st.pyplot(fig2)
    st.warning(f"Predictive Inference: Using **{mat_upright}** uprights with **{brake_disk}** rotors. Thermal expansion at 800°C will induce {round(camber_drift.max(), 3)}° of kinematic drift.")

with tabs[2]:
    st.header("Aero-Elastic CoP Shift & Stability Derivative")
    # Millimeters of Center of Pressure shift vs Velocity
    fig3, ax3 = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax3.plot(v_ax, cop_ax * 1000, color='#00ff9d', lw=3)
    ax3.set_xlabel("Velocity (km/h)"); ax3.set_ylabel("CoP Shift (mm Aft)")
    st.pyplot(fig3)
    st.markdown(f"**Stability Derivative:** At V-Max (360km/h), the **{wing_elements}** wing generates enough torque to shift the Center of Pressure **{round(cop_ax.max()*1000, 1)}mm** aft. Because it is **{aero_mount}**, this load will {'bypass the springs' if 'Suspension' in aero_mount else 'induce pitch-compression'}.")

with tabs[3]:
    st.header("🤖 Multi-Objective Chief Agent")
    subj = st.text_input("Jay, enter technical feedback (e.g. 'Front-end push at high-speed apex')")
    manifest = f"DNA: {mat_upright} Uprights, {wing_elements} Wing ({aero_mount}). Stagger: {f_width}/{r_width}. K: {round(k_val, 2)}."
    
    if q := st.chat_input("Inquire for engineering validation..."):
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                st.markdown(model.generate_content(f"{manifest}\n\nUSER: {q}").text)
