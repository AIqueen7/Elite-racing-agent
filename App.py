import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai
from PIL import Image

# --- CONFIGURATION & AGENT IDENTITY ---
st.set_page_config(page_title="Elite-Racing-Agent AI", page_icon="🤖", layout="wide")

# Championship UI Styling
st.markdown("""
    <style>
    .main { background-color: #050505; color: #ffffff; }
    [data-testid="stMetricValue"] { font-size: 38px !important; color: #00ff41; }
    .agent-header { padding: 10px; border-radius: 5px; background: #111; border-left: 5px solid #00ff41; margin-bottom: 20px; }
    .stChatInput { bottom: 20px; }
    .stButton>button { background-color: #00ff41; color: black; font-weight: bold; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE (AGENT MEMORY) ---
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225
if 'telemetry_summary' not in st.session_state: st.session_state['telemetry_summary'] = "No data uploaded yet."

# --- API SETUP ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

# --- SIDEBAR: SYSTEM PARAMETERS ---
with st.sidebar:
    st.title("🤖 AGENT CONFIG")
    weight = st.number_input("Vehicle Mass (kg)", value=850)
    power = st.number_input("Max Power (HP)", value=600)
    mu = st.slider("Surface Friction (μ)", 0.5, 2.0, 1.2)
    track = st.text_input("Track Location", value="Strawberry Creek Raceway")
    
    if st.button("SYNC ENVIRONMENTAL SENSORS"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat=53.3377&lon=-114.1603&appid={WEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            temp_k = res['main']['temp'] + 273.15
            st.session_state['rho'] = round((res['main']['pressure']*100) / (287.05 * temp_k), 4)
            st.success("Environment Synced.")
        except: st.error("Weather API Offline.")

# --- AGENT LOGIC: DATA INTERPRETATION ---
rho = st.session_state['rho']
p_to_w = power / weight
prob = min(99, max(10, int((p_to_w * (rho/1.225) / 0.75) * 100)))

# --- MAIN LAYOUT: MULTI-MODAL AGENT ---
st.markdown('<div class="agent-header"><b>AGENT STATUS: ACTIVE</b> | Location: Strawberry Creek | System: Digital Twin v3.0</div>', unsafe_allow_html=True)

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("WIN PROBABILITY", f"{prob}%")
m2.metric("AIR DENSITY (ρ)", f"{rho}")
m3.metric("DA ESTIMATE", f"{int((1.225-rho)*10000)} ft")
m4.metric("PWR/WGT", f"{round(p_to_w, 2)}")

tab1, tab2 = st.tabs(["📊 Telemetry & Visuals", "💬 Engineer Chat"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Performance Map")
        v_kmh = np.linspace(10, 250, 100)
        v_ms = v_kmh / 3.6
        # Traction-Limited Acceleration Formula: $a = \min(\mu, \frac{P}{v \cdot m \cdot g})$
        a_g = [min(((power*745.7/v) - (0.5*rho*v**2*0.45*1.5)) / (weight*9.81), mu) for v in v_ms]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.style.use('dark_background')
        ax.plot(v_kmh, a_g, color='#00ff41', linewidth=3, label="Digital Twin")
        
        up_file = st.file_uploader("Upload Session Data (CSV)", type="csv")
        if up_file:
            df = pd.read_csv(up_file)
            if 'speed' in df.columns and 'accel' in df.columns:
                ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=10, alpha=0.5)
                # Agent automatically summarizes the data
                st.session_state['telemetry_summary'] = f"Max Speed: {df['speed'].max()} kmh, Max G: {df['accel'].max()}G."
        
        ax.set_ylim(0, 1.6); ax.legend(); st.pyplot(fig)

    with col2:
        st.subheader("G-G Friction")
        fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
        circle = plt.Circle((0, 0), mu, color='#00ff41', fill=False, linestyle='--')
        ax_gg.add_artist(circle)
        if up_file and 'lat_g' in df.columns:
            ax_gg.scatter(df['lat_g'], df['accel'], color='white', s=5, alpha=0.3)
        ax_gg.set_xlim(-1.5, 1.5); ax_gg.set_ylim(-1.5, 1.5)
        st.pyplot(fig_gg)

with tab2:
    st.subheader("Race Engineer Interface")
    
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask your Engineer..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if GOOGLE_API_KEY:
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # The "Agent Context" - feeding the AI all current data
                agent_context = f"""
                You are the 'Elite-Racing-Agent'. You are a witty, professional Championship Lead Engineer.
                Current Setup: {power}HP, {weight}kg, {mu} tire grip.
                Environment: {track}, Air Density {rho}.
                Telemetry Data: {st.session_state['telemetry_summary']}
                User asks: {prompt}
                Give a tactical, high-performance response.
                """
                response = model.generate_content(agent_context)
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            else:
                st.error("AI Offline: Add GOOGLE_API_KEY to Secrets.")

st.caption("Elite-Racing-Agent v3.0 | AI-Co-Pilot Active")
