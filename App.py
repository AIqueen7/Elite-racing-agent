import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# --- 1. PRO-SPEC ARCHITECTURE ---
st.set_page_config(page_title="Elite-Racing-Agent | Master Builder Spec", page_icon="🏁", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #d1d1d1; }
    [data-testid="stMetricValue"] { font-size: 36px !important; color: #00e5ff; font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #111; color: #666; border: 1px solid #222; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #ffffff; border-bottom: 2px solid #00e5ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GLOBAL TRACK DATABASE ---
TRACKS = {
    "Strawberry Creek Raceway (Home)": {"lat": 53.3377, "lon": -114.1603, "alt": 2300},
    "Pikes Peak - Start": {"lat": 38.8405, "lon": -104.9442, "alt": 9390},
    "Pikes Peak - Summit": {"lat": 38.8405, "lon": -105.0445, "alt": 14115},
    "Nürburgring Nordschleife": {"lat": 50.3341, "lon": 6.9427, "alt": 2000},
    "Laguna Seca": {"lat": 36.5841, "lon": -121.7533, "alt": 800},
    "Mount Panorama (Bathurst)": {"lat": -33.4475, "lon": 149.559, "alt": 2800}
}

# --- 3. PHYSICS & SENSITIVITY ENGINES ---
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

# --- 4. SIDEBAR INPUTS ---
if 'rho' not in st.session_state: st.session_state['rho'] = 1.225

with st.sidebar:
    st.title("🛠️ CHASSIS LAB")
    venue_key = st.selectbox("Track Selection", list(TRACKS.keys()))
    v_data = TRACKS[venue_key]
    
    with st.expander("Mechanical Parameters", expanded=True):
        hp = st.number_input("Base BHP", 100, 2500, 600)
        kg = st.number_input("Mass (kg)", 500, 3000, 850)
        mu = st.slider("Mechanical Grip (μ)", 0.5, 2.5, 1.4)
        cd = st.slider("Aero Drag (Cd)", 0.1, 1.5, 0.45)

    if st.button("SYNC REMOTE ATMOSPHERE"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={v_data['lat']}&lon={v_data['lon']}&appid={st.secrets.get('OPENWEATHER_API_KEY')}&units=metric"
            res = requests.get(url).json()
            tk = res['main']['temp'] + 273.15
            st.session_state['rho'] = round((res['main']['pressure']*100)/(287.05*tk), 4)
            st.success("Atmosphere Locked")
        except: st.error("Sensor Sync Failed")

rho = st.session_state['rho']
v_ref = np.linspace(5, 320, 100)
curve, cur_hp = sim_physics(hp, kg, rho, cd, mu, v_ref)

# --- 5. THE MASTER DASHBOARD ---
t1, t2, t3 = st.tabs(["📊 PERFORMANCE SUMMARY", "🧬 PHYSICS & SENSITIVITY", "🤖 CHIEF AGENT"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    da = int((1.225 - rho) * 10000)
    c1.metric("DENSITY ALTITUDE", f"{da} ft")
    c2.metric("EFFECTIVE BHP", f"{int(cur_hp)}")
    c3.metric("AIR DENSITY (ρ)", f"{rho}")
    vmax = int(np.cbrt((cur_hp * 745.7 * 0.85) / (0.5 * rho * cd * 1.5)) * 3.6)
    c4.metric("REAL V-MAX", f"{vmax} kmh", f"{int(vmax*0.621)} mph")

    mc, sc = st.columns([2, 1])
    with mc:
        st.subheader("Longitudinal G-Envelope")
        fig, ax = plt.subplots(figsize=(10, 4.5)); plt.style.use('dark_background')
        ax.plot(v_ref, curve, color='#00e5ff', lw=2.5, label="Physics Model")
        f = st.file_uploader("Upload Telemetry (CSV)", type="csv")
        if f:
            df = pd.read_csv(f)
            if 'speed' in df.columns and 'accel' in df.columns:
                ax.scatter(df['speed'], df['accel'], c=df['accel'], cmap='magma', s=10, alpha=0.4)
        ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("G-Force"); ax.legend(); st.pyplot(fig)

    with sc:
        st.subheader("G-G Friction Circle")
        fig_gg, ax_gg = plt.subplots(figsize=(5, 5))
        t = np.linspace(0, 2*np.pi, 100)
        ax_gg.plot(mu*np.cos(t), mu*np.sin(t), color='#00e5ff', ls='--', alpha=0.4)
        if f and 'lat_g' in df.columns and 'accel' in df.columns:
            ax_gg.scatter(df['lat_g'], df['accel'], color='white', s=3, alpha=0.3)
            g_u = np.sqrt(df['lat_g']**2 + df['accel']**2).mean()
            st.write(f"Mean Grip Usage: {round((g_u/mu)*100,1)}%")
        ax_gg.set_xlim(-mu-0.2, mu+0.2); ax_gg.set_ylim(-mu-0.2, mu+0.2); st.pyplot(fig_gg)

with t2:
    st.header("Deep Chassis Analysis")
    g1, g2 = st.columns(2)
    
    with g1:
        # GRAPH 1: Aero vs Mechanical Grip Shift
        st.subheader("Grip Source Correlation")
        aero_grip = [0.5 * rho * (v/3.6)**2 * (cd * 2.0) / (kg * 9.81) for v in v_ref]
        fig1, ax1 = plt.subplots(); plt.style.use('dark_background')
        ax1.fill_between(v_ref, mu, color='#333', label="Mechanical Limit")
        ax1.plot(v_ref, aero_grip, color='#00e5ff', lw=2, label="Aero Component")
        ax1.set_xlabel("Speed (km/h)"); ax1.set_ylabel("G-Potential"); ax1.legend(); st.pyplot(fig1)

    with g2:
        # GRAPH 2: Tire Energy Flux (Work Done)
        st.subheader("Tire Energy Budget")
        if f and 'g_sum' in df.columns:
            energy = df['g_sum'] * df['speed']
            fig2, ax2 = plt.subplots(); plt.style.use('dark_background')
            ax2.plot(df['timestamp'], energy, color='#ff4b4b')
            ax2.set_ylabel("Energy Load (Work)"); st.pyplot(fig2)
        else:
            st.info("Upload CSV with 'g_sum' to see Tire Energy flux.")

    g3, g4 = st.columns(2)
    
    with g3:
        # GRAPH 3: Altitude BHP Decay (Pikes Peak Specific)
        st.subheader("The Altitude Tax")
        alts = np.linspace(0, 15000, 50)
        p_c = [hp * (((1.225 * np.exp(-a / 30000)) / 1.225) ** 0.7) for a in alts]
        fig3, ax3 = plt.subplots(); plt.style.use('dark_background')
        ax3.plot(alts, p_c, color='#ff8700', lw=2)
        ax3.axvline(v_data['alt'], color='white', ls='--')
        ax3.set_xlabel("Altitude (ft)"); ax3.set_ylabel("BHP"); st.pyplot(fig3)

    with g4:
        # GRAPH 4: Downforce vs Drag Penalty
        st.subheader("Aero Efficiency (L/D)")
        drag = [0.5 * rho * (v/3.6)**2 * cd for v in v_ref]
        downforce = [d * 2.5 for d in drag] # Simple Lift/Drag ratio
        fig4, ax4 = plt.subplots(); plt.style.use('dark_background')
        ax4.plot(v_ref, downforce, color='#00e5ff', label="Downforce (N)")
        ax4.plot(v_ref, drag, color='#ff4b4b', label="Drag (N)")
        ax4.set_xlabel("Speed (km/h)"); ax4.legend(); st.pyplot(fig4)

with t3:
    q = st.chat_input("Technical inquiry...")
    if q:
        with st.chat_message("assistant"):
            if st.secrets.get("GOOGLE_API_KEY"):
                genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
                m = genai.GenerativeModel('gemini-1.5-flash')
                ctx = f"Consultant for Jay Esterer (Unlimited Class Racer). {hp}HP, {kg}kg. Track: {venue_key}. Query: {q}"
                st.write(m.generate_content(ctx).text)

st.caption(f"v10.0 | Master Builder Spec | Total Signal Integration")
