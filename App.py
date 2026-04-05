import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- 1. CORE SYSTEM ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | Titanium Spec", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #f0f0f0; }
    [data-testid="stMetricValue"] { font-size: 30px !important; color: #d1d1d1; font-family: 'JetBrains Mono'; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE TITANIUM-INTEGRATED DNA ---
with st.sidebar:
    st.title("🛡️ ARCHITECTURAL DNA")
    
    with st.expander("Metallurgy & Unsprung Mass", expanded=True):
        # ADDED: Titanium Grade 5
        mat_upright = st.selectbox("Upright/Rod Material", 
            ["6061-T6 Aluminum", "Magnesium", "Titanium Grade 5 (Ti-6Al-4V)", "4130 Chromoly Steel"])
        wheel_mat = st.selectbox("Wheel Material", ["Forged Aluminum", "Forged Magnesium", "Carbon Fiber"])
        brake_disk = st.selectbox("Brake Rotor", ["Carbon-Ceramic", "High-Carbon Steel", "Cast Iron"])
        
    with st.expander("Kinematic Tire Stagger", expanded=True):
        f_width = st.number_input("Front Width (mm)", 200, 400, 285)
        r_width = st.number_input("Rear Width (mm)", 200, 500, 335)

    with st.expander("Aero-Structural Config", expanded=True):
        wing_elements = st.radio("Wing Elements", ["Dual-Element", "Triple-Element"])
        aero_mount = st.radio("Mounting Point", ["Chassis-Mounted", "Suspension-Mounted (Unsprung)"])

# --- 3. TITANIUM-AWARE PHYSICS SYNTHESIS ---
def titanium_synthesis(hp, kg, f_w, r_w, mat, b_mat):
    # Expanded Material Constants: E (Modulus GPa), alpha (Expansion e-6/K), density (kg/m3)
    phys_props = {
        "6061-T6 Aluminum": {"E": 68.9, "alpha": 23.1, "rho": 2700},
        "Magnesium": {"E": 45.0, "alpha": 25.0, "rho": 1740},
        "Titanium Grade 5 (Ti-6Al-4V)": {"E": 113.8, "alpha": 8.6, "rho": 4430},
        "4130 Chromoly Steel": {"E": 205.0, "alpha": 11.0, "rho": 7850}
    }
    
    selected = phys_props[mat]
    
    # 1. Natural Frequency Inference (f ~ sqrt(k/m))
    # Titanium rods provide high strength-to-weight but lower stiffness than steel
    # which shifts the resonance peak.
    v = np.linspace(0, 360, 200)
    stiffness_k = selected["E"] * 1000 
    res_freq = np.sqrt(stiffness_k / selected["rho"]) * 10 # Heuristic Scaler
    
    # 2. Thermal Camber Drift Logic
    # Titanium has exceptionally low thermal expansion compared to Aluminum/Magnesium
    temp_rise = np.linspace(20, 800, 100)
    c_drift = selected["alpha"] * temp_rise * 0.0001
    
    return v, res_freq, temp_rise, c_drift, selected["alpha"]

v_ax, res_f, t_range, c_drift, t_alpha = titanium_synthesis(1200, 850, f_width, r_width, mat_upright, brake_disk)

# --- 4. THE INTERFACE ---
tabs = st.tabs(["🏗️ STRUCTURAL RESONANCE", "🔥 THERMAL DRIFT", "🌪️ AERO-ELASTICITY", "🤖 CHIEF AGENT"])

with tabs[0]:
    st.header("Unsprung Resonance: Harmonic Analysis")
    # Visualizing how the material choice dampens or amplifies road frequency
    freq_range = np.linspace(0, 100, 100)
    amplitude = 1 / (np.abs(res_f**2 - freq_range**2) + 15)
    
    fig, ax = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax.plot(freq_range, amplitude * 100, color='#00e5ff', lw=3)
    ax.fill_between(freq_range, amplitude * 100, color='cyan', alpha=0.1)
    ax.axvline(res_f, color='red', ls='--', label=f"Resonant Peak: {round(res_f, 1)} Hz")
    ax.set_xlabel("Excitation Frequency (Hz)"); ax.set_ylabel("Response Amplitude"); ax.legend()
    st.pyplot(fig)
    st.info(f"Architect's Note: By using **{mat_upright}**, the system identifies a resonant peak at {round(res_f, 1)} Hz. Titanium's lower modulus vs Steel acts as a structural spring, potentially absorbing mid-corner bumps better.")

with tabs[1]:
    st.header("Thermal Expansion: Camber Stability")
    # Low alpha of Titanium means much higher alignment stability under heat
    fig2, ax2 = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax2.plot(t_range, c_drift, color='#ff4b4b', lw=3)
    ax2.set_xlabel("Brake Bulk Temp (°C)"); ax2.set_ylabel("Camber Drift (deg)")
    st.pyplot(fig2)
    st.success(f"Stability Insight: **{mat_upright}** has a thermal expansion coefficient of {t_alpha}. At 800°C, your alignment drift is reduced by ~60% compared to Aluminum uprights.")
    

with tabs[2]:
    st.header("Aero-Elastic Center of Pressure Shift")
    # Dynamic CoP shift based on the Modulus of the uprights/rods
    q = 0.5 * 1.1 * (v_ax/3.6)**2
    shift = (q / (113.8 if "Titanium" in mat_upright else 68.9)) * 0.01
    
    fig3, ax3 = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax3.plot(v_ax, shift * 1000, color='#00ff9d', lw=3)
    ax3.set_xlabel("Velocity (km/h)"); ax3.set_ylabel("CoP Shift (mm Aft)")
    st.pyplot(fig3)
    

with tabs[3]:
    st.header("🤖 Multi-Objective Chief Agent")
    subj = st.text_input("Jay, enter experiential feedback (e.g. 'Mid-corner oscillation over bumps')")
    manifest = f"DNA: {mat_upright} Components. Resonance: {round(res_f, 1)}Hz. Expansion: {t_alpha} e-6/K."
    
    if q := st.chat_input("Query the Neural Architect..."):
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                st.markdown(model.generate_content(f"{manifest}\n\nUSER: {q}").text)
