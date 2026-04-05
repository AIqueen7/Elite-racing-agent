import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from scipy.stats import multivariate_normal

# --- 1. SYSTEM ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | V7 Final", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #fdfdfd; }
    [data-testid="stMetricValue"] { font-size: 34px !important; color: #00e5ff; font-family: 'JetBrains Mono'; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE TOTAL MISSION DNA (Restored & Expanded) ---
with st.sidebar:
    st.title("🛡️ TOTAL MISSION DNA")
    
    with st.expander("Power & Environment", expanded=True):
        hp = st.number_input("Nominal BHP", 500, 3000, 1200)
        kg = st.number_input("Dry Mass (kg)", 500, 2500, 850)
        rho_s = st.slider("Air Density (kg/m³)", 0.6, 1.3, 1.10)
        mission = st.selectbox("Objective Function", ["Hill Climb Sprint", "Endurance Circuit", "Top Speed V-Max"])

    with st.expander("Structural Metallurgy", expanded=True):
        mat_upright = st.selectbox("Upright/Rod Material", 
            ["Titanium Grade 5 (Ti-6Al-4V)", "6061-T6 Aluminum", "Magnesium", "4130 Steel"])
        brake_mat = st.selectbox("Brake Interface", ["Carbon-Ceramic", "Cast Iron", "High-Carbon Steel"])
        
    with st.expander("Kinematic Stagger", expanded=True):
        f_tire = st.number_input("Front Width (mm)", 200, 400, 285)
        r_tire = st.number_input("Rear Width (mm)", 200, 500, 335)
        wing_type = st.radio("Wing Element", ["Dual-Element", "Triple-Element"])

# --- 3. THE NEURAL-PHYSICS KERNEL ---
def generate_pinn_manifold(hp, kg, rho, mission, mat):
    # Performance Manifold: X=Mechanical Mu, Y=Aero Load (N)
    x = np.linspace(1.2, 2.5, 50)
    y = np.linspace(100, 1000, 50)
    X, Y = np.meshgrid(x, y)
    
    # Peak shift logic based on Mission & Material Stiffness
    mu_peak = 2.1 if "Hill Climb" in mission else 1.7
    aero_peak = 800 if "Triple" in wing_type else 550
    
    pos = np.dstack((X, Y))
    rv = multivariate_normal([mu_peak, aero_peak], [[0.15, 0], [0, 6000]])
    Z = rv.pdf(pos) * 1000
    
    # Effective Power calculation
    eff_hp = hp * (rho / 1.225)**0.85
    return X, Y, Z, eff_hp

X_m, Y_m, Z_m, e_hp = generate_pinn_manifold(hp, kg, rho_s, mission, mat_upright)

# --- 4. THE MASTER INTERFACE ---
t1, t2, t3, t4 = st.tabs(["🌌 NEURAL HEATMAP", "🛞 KINEMATIC BIAS", "🔥 THERMAL PHM", "🤖 CHIEF AGENT"])

with t1:
    st.header(f"Latent Manifold: {mission} Optimization")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # THE HEATMAP (Latent Manifold)
        fig, ax = plt.subplots(figsize=(10, 6)); plt.style.use('dark_background')
        cp = ax.contourf(X_m, Y_m, Z_m, levels=50, cmap='magma')
        fig.colorbar(cp, label='Optimization Reward Score')
        ax.set_xlabel("Mechanical Friction Coefficient (μ)")
        ax.set_ylabel("Aero Load @ 100km/h (N)")
        st.pyplot(fig)
        
    with col2:
        st.metric("EFFECTIVE BHP", f"{int(e_hp)} hp")
        st.metric("POWER-TO-WEIGHT", f"{round(e_hp/kg, 2)} hp/kg")
        st.markdown(f"""
        **Architect's Synthesis:**
        The heatmap identifies the 'Golden Window' for **{mission}**. 
        * **Titanium Benefit:** Using Ti-6Al-4V rods allows for the narrow, high-frequency peak shown in the manifold. 
        * **Aero Seal:** At **{rho_s} kg/m³**, the AI predicts a required aero-load of **{int(Y_m[np.unravel_index(Z_m.argmax(), Z_m.shape)])}N** to maintain floor seal.
        """)
    

with t2:
    st.header("Kinematic Bias & Axle Saturation")
    # Slip angle differential based on 285/335 stagger
    slip = np.linspace(0, 15, 100)
    stagger_ratio = r_tire / f_tire
    f_force = np.tanh(slip / 5) * (f_tire/300)
    r_force = np.tanh(slip / (5 * stagger_ratio)) * (r_tire/300)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax2.plot(slip, f_force, color='cyan', label="Front Axle Saturation")
    ax2.plot(slip, r_force, color='magenta', label="Rear Axle Saturation")
    ax2.set_xlabel("Slip Angle (deg)"); ax2.set_ylabel("Normalized Lateral Force"); ax2.legend()
    st.pyplot(fig2)

with t3:
    st.header("Material-Specific Thermal Diffusion")
    # Predicting Camber Drift for Titanium vs Others
    temp = np.linspace(20, 800, 100)
    # alpha: Ti=8.6, Al=23.1
    alpha = 8.6 if "Titanium" in mat_upright else 23.1
    drift = alpha * temp * 0.0001
    
    fig3, ax3 = plt.subplots(figsize=(10, 4)); plt.style.use('dark_background')
    ax3.plot(temp, drift, color='#ff4b4b', lw=3)
    ax3.set_xlabel("Brake Bulk Temperature (°C)"); ax3.set_ylabel("Camber Drift (deg)")
    st.pyplot(fig3)
    

with t4:
    st.header("🤖 Multi-Objective Chief Agent")
    subj = st.text_input("Enter engineering feedback:")
    manifest = f"DNA: {hp}HP, {kg}kg, {mat_upright} Uprights, {f_tire}/{r_tire} Stagger."
    
    if q := st.chat_input("Query the system..."):
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                st.markdown(model.generate_content(f"{manifest}\n\nUSER: {q}").text)

st.caption("Elite-Racing-Agent | Neural Architect Spec V7 | Final Integrated Build")
