import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. PRO-SPEC ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | Summit Spec", page_icon="🏁", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #070707; color: #d1d1d1; }
    [data-testid="stMetricValue"] { font-size: 38px !important; color: #00e5ff; font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #111; color: #666; border: 1px solid #222; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #ffffff; border-bottom: 2px solid #00e5ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TRACK & PHYSICS ENGINE ---
TRACKS = {
    "Strawberry Creek (Home)": {"lat": 53.3377, "lon": -114.1603, "alt": 2300},
    "Pikes Peak - Start": {"lat": 38.8405, "lon": -104.9442, "alt": 9390},
    "Pikes Peak - Summit": {"lat": 38.8405, "lon": -105.0445, "alt": 14115},
    "Mt. Washington": {"lat": 44.2705, "lon": -71.3033, "alt": 6288},
    "Custom Location": {"lat": 0, "lon": 0, "alt": 0}
}

def sim_physics(power, weight, rho, cd, mu, v_range):
    v_ms = v_range / 3.6
    eff_hp = power * ((rho / 1.225) ** 0.7)
    p_w = eff_hp * 745.7
    gs = []
    for v in v_ms:
        v = max(v, 1.0)
        drag = 0.5 * rho * (v**2) * cd * 1.5
        net_f = ((p_w / v) * 0.85) - drag - (weight * 9.81 * 0.015)
        gs.append(max(min(net_f / (weight * 9.81), mu), -mu))
    return gs, eff_hp

# --- 3. SIDEBAR & INPUTS ---
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🏔️ MISSION CONFIG")
    venue = st.selectbox("Location Selection", list(TRACKS.keys()))
    v_data = TRACKS[venue]
    
    with st.expander("Chassis DNA", expanded=True):
        hp = st.number_input("Sea Level BHP", 100, 2500, 600)
        kg = st.number_input("Race Mass (kg)", 500, 3000, 850)
        mu = st.slider("Grip (μ)", 0.5, 2.5, 1.4)
        cd = st.slider("Drag (Cd)", 0.2, 1.2, 0.45)

    if st.button("SYNC ATMOSPHERE"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={v_data['lat']}&lon={v_data['lon']}&appid={st.secrets.get('OPENWEATHER_API_KEY')}&units=metric"
            res = requests.get(url).json()
            tk = res['main']['temp'] + 273.15
            st.session_state['rho'] = round((res['main']['pressure']*100)/(287.05*tk), 4)
            st.success("Synced")
        except: st.error("API Error")

rho = st.session_state['rho']
v_ref = np.linspace(5, 300, 100)
curve, cur_hp = sim_physics(hp, kg, rho, cd, mu, v_ref)

# --- 4. DASHBOARD ---
t1, t2, t3 = st.tabs(["📊 TELEMETRY", "🧬 ALTITUDE", "🤖 AGENT"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    da = int((1.225 - rho) * 10000)
    c1.metric("DENSITY ALTITUDE", f"{da} ft")
    c2.metric("EFFECTIVE BHP", f"{int(cur_hp)}")
    c3.metric("AIR DENSITY", f"{rho}")
    vmax = int(np.cbrt((cur_hp * 745.7 * 0.85) / (0.5 * rho * cd * 1.5)) * 3.6)
    c4.metric("REAL V-MAX", f"{vmax} kmh", f"{int(vmax*0.621)} mph")

    mc, sc = st.columns([2, 1])
    with mc:
        st.subheader(f"Envelope: {venue}")
        fig, ax = plt.subplots(figsize=(10, 4.5)); plt.style.use('dark_background')
        ax.plot(v_ref, curve, color='#00e5ff', lw=2, label="Theoretical")
        f = st.file_uploader("Upload CSV", type="csv")
        if f:
            df = pd.read_csv(f)
            if 'speed' in df.columns and 'accel' in df.columns:
                ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=8, alpha=0.5)
        ax.set_xlabel("Speed"); ax.set_ylabel("G"); ax.legend(); st.pyplot(fig)

    with sc:
        st.subheader("G-G Circle")
        fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
        t = np.linspace(0, 2*np.pi, 100)
        ax_gg.plot(mu*np.cos(t), mu*np.sin(t), color='#00e5ff', ls='--', alpha=0.5)
        if f and 'lat_g' in df.columns and 'accel' in df.columns:
            ax_gg.scatter(df['lat_g'], df['accel'], color='white', s=3, alpha=0.4)
            g_u = np.sqrt(df['lat_g']**2 + df['accel']**2).mean()
            st.write(f"Grip Usage: {round((g_u/mu)*100,1)}%")
        ax_gg.set_xlim(-mu-0.2, mu+0.2); ax_gg.set_ylim(-mu-0.2, mu+0.2); st.pyplot(fig_gg)

with t2:
    st.header("Altitude Sensitivity")
    alts = np.linspace(0, 15000, 50)
    dens = 1.225 * np.exp(-alts / 30000)
    p_c = [hp * ((d / 1.225) ** 0.7) for d in dens]
    fig_a, ax_a = plt.subplots(figsize=(10, 4))
    ax_a.plot(alts, p_c, color='#ff4b4b', lw=3)
    if venue != "Custom Location": ax_a.axvline(v_data['alt'], color='white', ls='--')
    ax_a.set_xlabel("Altitude (ft)"); ax_a.set_ylabel("BHP"); st.pyplot(fig_a)

with t3:
    q = st.chat_input("Technical inquiry...")
    if q:
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                m = genai.GenerativeModel('gemini-1.5-flash')
                ctx = f"Engineer for Jay Esterer. {hp}HP. {venue}. {q}"
                st.write(m.generate_content(ctx).text)

st.caption(f"v8.6 | Summit Spec | {venue}")
