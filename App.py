import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. SYSTEM ARCHITECTURE & STYLING ---
st.set_page_config(page_title="Elite-Racing-Agent | Pro Spec", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #e0e0e0; }
    [data-testid="stMetricValue"] { font-size: 42px !important; color: #ff8700; font-family: 'Courier New'; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; background-color: #111; border-radius: 5px 5px 0 0; color: white; border: 1px solid #333;
    }
    .status-banner { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 20px; }
    .science-card { background: #0a0a0a; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE PHYSICS CORE ---
def calculate_accel(power, weight, rho, cd, mu, v_range):
    v_ms = v_range / 3.6
    p_watts = power * 745.7
    return [min(((p_watts/v) - (0.5*rho*v**2*cd*1.5)) / (weight*9.81), mu) for v in v_ms]

def run_stochastic_sim(power, weight, rho, cd, mu, iterations=400):
    results = []
    for _ in range(iterations):
        n_rho = np.random.normal(rho, rho * 0.04)
        n_mu = np.random.normal(mu, mu * 0.06)
        val = calculate_accel(power, weight, n_rho, cd, n_mu, np.array([100]))
        results.append(val[0])
    return results

# --- 3. SESSION & API CONFIG ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

# --- 4. SIDEBAR: DATA INPUTS ---
with st.sidebar:
    st.title("🛠️ TECH SPEC")
    track_name = st.text_input("Track Identifier", "Strawberry Creek Raceway")
    
    with st.expander("Vehicle DNA", expanded=True):
        power = st.number_input("Horsepower (HP)", 100, 2000, 600)
        weight = st.number_input("Mass (kg)", 500, 2500, 850)
        mu = st.slider("Tire Grip (μ)", 0.5, 2.0, 1.2)
        cd = st.slider("Aero Drag (Cd)", 0.2, 0.9, 0.45)

    with st.expander("Environmental Sync"):
        if st.button("PULL LIVE DATA"):
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?lat=53.3377&lon=-114.1603&appid={WEATHER_API_KEY}&units=metric"
                res = requests.get(url).json()
                t_k = res['main']['temp'] + 273.15
                st.session_state['rho'] = round((res['main']['pressure']*100) / (287.05 * t_k), 4)
                st.success("Synced to Track Barometer")
            except: st.error("Weather API Timeout")
        rho = st.session_state['rho']

# --- 5. DASHBOARD CALCULATIONS ---
sim_data = run_stochastic_sim(power, weight, rho, cd, mu)
mean_g = np.mean(sim_data)
stability_score = 100 - (np.std(sim_data) / (mean_g if mean_g != 0 else 1) * 100)
win_prob = int((mean_g / mu) * 100) if mu != 0 else 0

# --- 6. MISSION CONTROL UI ---
tab_dashboard, tab_science, tab_agent, tab_roi = st.tabs(["🏎️ MISSION CONTROL", "🔬 SCIENCE LAB", "🤖 AGENT COMMS", "💰 MCLAREN ROI"])

with tab_dashboard:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WIN PROBABILITY", f"{win_prob}%")
    m2.metric("STABILITY INDEX", f"{round(stability_score, 1)}", "68% Conf")
    m3.metric("AIR DENSITY", f"{rho}")
    m4.metric("POWER/WEIGHT", f"{round(power/weight, 2)}")

    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.subheader("Performance Envelope: Target vs Actual")
        v_range = np.linspace(10, 260, 100)
        target = calculate_accel(power, weight, rho, cd, mu, v_range)
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.style.use('dark_background')
        ax.plot(v_range, target, color='#ff8700', linewidth=3, label="Digital Twin")
        
        uploaded = st.file_uploader("Upload Session Telemetry (CSV)", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if 'speed' in df.columns and 'accel' in df.columns:
                color_map = df['hr'] if 'hr' in df.columns else df['accel']
                label_txt = "HR (BPM)" if 'hr' in df.columns else "G-Force"
                scat = ax.scatter(df['speed'], df['accel'], c=color_map, cmap='magma', s=12, alpha=0.6)
                plt.colorbar(scat, label=label_txt)
        ax.set_ylim(0, mu + 0.3); ax.legend(); st.pyplot(fig)

    with c_right:
        st.subheader("G-G Stability Circle")
        fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
        circle = plt.Circle((0, 0), mu, color='#ff8700', fill=False, linestyle='--')
        ax_gg.add_artist(circle)
        if uploaded:
            df = pd.read_csv(uploaded)
            if 'lat_g' in df.columns and 'accel' in df.columns:
                ax_gg.scatter(df['lat_g'], df['accel'], color='white', s=3, alpha=0.3)
        ax_gg.set_xlim(-mu-0.2, mu+0.2); ax_gg.set_ylim(-mu-0.2, mu+0.2)
        st.pyplot(fig_gg)

with tab_science:
    st.header("Stochastic Performance Modeling")
    fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
    plt.style.use('dark_background')
    ax_mc.hist(sim_data, bins=35, color='#ff8700', alpha=0.5, label="Simulated Runs")
    ax_mc.axvline(mean_g, color='white', linestyle='--', label=f"Mean: {round(mean_g, 2)}G")
    ax_mc.axvspan(mean_g - np.std(sim_data), mean_g + np.std(sim_data), color='#ff8700', alpha=0.1, label="1-Sigma Confidence")
    ax_mc.set_title("Stochastic Probability Distribution")
    ax_mc.legend()
    st.pyplot(fig_mc)

    st.markdown(r"""
    #### The Math Behind the G-Force
    The acceleration potential is the minimum of tire friction ($\mu$) and available power force:
    $$ a = \min \left( \mu, \frac{\frac{P}{v} - \frac{1}{2}\rho v^2 C_d A}{m \cdot g} \right) $$
    """)

with tab_agent:
    st.subheader("Race Engineer Comms")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if p := st.chat_input("Query the Agent..."):
        st.session_state.chat_history.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            if GOOGLE_API_KEY:
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context = f"Racing Engineer. Setup: {power}HP, {weight}kg, {rho} density. Track: {track_name}. User says: {p}"
                resp = model.generate_content(context).text
                st.markdown(resp)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
            else: st.error("AI Key Missing")

with tab_roi:
    st.header("Component ROI Optimizer")
    st.info("Calculate the Cost-Per-Tenth for new vehicle components.")
    c1, c2 = st.columns(2)
    part_cost = c1.number_input("Component Cost ($)", value=5000)
    time_gain = c2.number_input("Estimated Lap Time Gain (seconds)", value=0.15)
    if time_gain > 0:
        cost_per_ms = part_cost / (time_gain * 1000)
        st.metric("COST PER MILLISECOND", f"${round(cost_per_ms, 2)}")
        if cost_per_ms < 50:
            st.success("Recommendation: ACQUIRE (High Efficiency)")
        else:
            st.warning("Recommendation: AUDIT (Low Efficiency Gain)")

st.caption(f"Elite-Racing-Agent v5.2 | McLaren Spec | {track_name}")
