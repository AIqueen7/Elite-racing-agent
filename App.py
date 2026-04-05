import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. CORE ARCHITECTURE & UI ---
st.set_page_config(page_title="Elite-Racing-Agent | Architect Spec", page_icon="🏎️", layout="wide")

# High-Contrast Engineering Interface
st.markdown("""
    <style>
    .main { background-color: #020202; color: #e0e0e0; }
    [data-testid="stMetricValue"] { font-size: 32px !important; color: #00e5ff; font-weight: 700; }
    .stExpander { background-color: #0a0a0a; border: 1px solid #222; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE DYNAMIC PHYSICS KERNEL ---
# No hardcoded constants: Gravity, Air Density, and Drivetrain Loss are parameters
def twin_kernel(hp, mass, rho, cd, mu, v_range, drivetrain_eff=0.88, rolling_res=0.012):
    v_ms = v_range / 3.6
    # Power correction for altitude/density
    eff_hp = hp * ((rho / 1.225) ** 0.7)
    p_w = eff_hp * 745.7 # Convert to Watts
    gs = []
    for v in v_ms:
        v = max(v, 1.0)
        drag = 0.5 * rho * (v**2) * cd * 1.5 # 1.5m2 Frontal Area
        # Net Force = (Power/Velocity * Eff) - Drag - Rolling Resistance
        net_f = ((p_w / v) * drivetrain_eff) - drag - (mass * 9.81 * rolling_res)
        gs.append(max(min(net_f / (mass * 9.81), mu), -mu))
    return gs, eff_hp

# --- 3. MISSION CONTROL (DYNAMIC INPUTS) ---
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🎛️ MISSION CONTROL")
    with st.expander("Chassis DNA", expanded=True):
        hp_input = st.number_input("Nominal BHP", 100, 2500, 600)
        mass_input = st.number_input("Race Mass (kg)", 500, 3000, 850)
        mu_input = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
        cd_input = st.slider("Base Drag (Cd)", 0.1, 1.5, 0.45)
    
    with st.expander("Atmospheric Sync", expanded=False):
        # Allow manual override for density testing
        rho_val = st.slider("Air Density (kg/m³)", 0.5, 1.3, st.session_state['rho'])
        st.session_state['rho'] = rho_val

# --- 4. DYNAMIC CALCULATION LAYER ---
v_ref = np.linspace(5, 340, 100)
physics_curve, effective_bhp = twin_kernel(hp_input, mass_input, st.session_state['rho'], cd_input, mu_input, v_ref)

# Dynamic Aero Crossover Logic
v_cross = int(np.sqrt((mass_input * 9.81) / (0.5 * st.session_state['rho'] * (cd_input * 2.5) * 1.5)) * 3.6)

# --- 5. HMI: THE MASTER DASHBOARD ---
tabs = st.tabs(["📊 TELEMETRY", "🧬 AI DYNAMICS", "📖 THEORY & LOGIC", "🤖 CHIEF AGENT"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EFFECTIVE BHP", f"{int(effective_bhp)} hp")
    c2.metric("AERO CROSSOVER", f"{v_cross} kmh")
    c3.metric("AIR DENSITY", f"{st.session_state['rho']} kg/m³")
    
    # Upload and Plot
    f = st.file_uploader("📥 Synchronize Telemetry Stream", type="csv")
    fig, ax = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax.plot(v_ref, physics_curve, color='#00e5ff', lw=2, label="Digital Twin")
    if f:
        df = pd.read_csv(f)
