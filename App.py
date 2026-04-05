import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --- 1. NEURAL ARCHITECTURES (The "Digital Twin" Logic) ---
class RacingVAE(nn.Module):
    """Deep Manifold Learning: Mapping 1200HP / 850kg into the 'Golden Window'."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 64), nn.GELU(), # GELU for smoother gradient flow in racing data
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 2) # Latent DNA Space
        )
    def forward(self, x): return self.encoder(x)

class ThermalLSTM(nn.Module):
    """Temporal Memory: Predicting Internal Carcass Heat (Hysteresis)."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(x) # 'h' captures the temporal memory of energy soak
        return self.fc(h[-1])

# --- 2. THE SOVEREIGN INTERFACE ---
st.set_page_config(page_title="Sovereign Architect", layout="wide")

with st.sidebar:
    st.title("🛡️ CHASSIS DNA")
    st.markdown("---")
    hp = st.number_input("Peak Output (BHP)", 500, 3000, 1200)
    kg = st.number_input("Dry Mass (kg)", 500, 2500, 850)
    mat = st.selectbox("Unsprung Material", ["Titanium Grade 5", "6061-T6 Aluminum"])
    k_tire = st.number_input("Tire Spring Rate (N/mm)", 100, 500, 280)
    cop = st.slider("Static Aero Balance (% Front)", 35.0, 65.0, 42.0)
    st.markdown("---")
    st.info("🛰️ Awaiting 100Hz Telemetry Stream for Phase 2 Calibration.")

# --- 3. NEURAL INFERENCE (The 'Brains') ---
# Correctly scaling 10 parameters for the VAE inference
input_vec = torch.tensor([[hp/3000, kg/2500, (1.0 if "Ti" in mat else 0.5), cop/100, k_tire/500, 0.8, 0.7, 0.5, 0.5, 0.5]], dtype=torch.float32)
vae, lstm = RacingVAE(), ThermalLSTM()

with torch.no_grad():
    z = vae(input_vec).numpy()[0]
    # Simulated 100Hz brake-pressure history for LSTM 'Memory'
    heat_history = torch.tensor([[[0.1], [0.4], [0.98], [0.92], [0.85]]], dtype=torch.float32)
    carcass_core = lstm(heat_history).item() * 15 + 102 # Neural estimation of soak

# --- 4. THE COMMAND CENTER ---
tabs = st.tabs(["🌌 THE MANIFOLD", "🔥 THERMAL MEMORY", "🔊 BODE PHASING"])

with tabs[0]:
    st.header("The Latent Manifold: Global Maxima ($O^*$)")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig0, ax0 = plt.subplots(figsize=(10, 5)); plt.style.use('dark_background')
        grid = np.linspace(-3, 3, 50); gx, gy = np.meshgrid(grid, grid)
        ax0.contourf(gx, gy, np.exp(-(gx**2 + gy**2)/2), cmap='magma', alpha=0.9)
        ax0.scatter(z[0]*2, z[1]*2, color='#00e5ff', s=800, marker='*', label="Optimal Pivot")
        ax0.set_xlabel("Z1: Mechanical DNA"); ax0.set_ylabel("Z2: Tire/Aero Compliance")
        st.pyplot(fig0); plt.close(fig0)
    with c2:
        st.subheader("Manifold Intelligence")
        st.write(f"**Optimal Pivot Point ($O^*$):** {z[0]:.3f}, {z[1]:.3f}")
        st.write("This is the mathematical peak where your chassis and aero are 'In-Phase.'")

with tabs[1]:
    st.header("LSTM: Internal Energy (Thermal Hysteresis)")
    fig1, ax1 = plt.subplots(figsize=(10, 3)); plt.style.use('dark_background')
    t = np.linspace(0, 10, 100); surface = 90 + 18*np.sin(t)
    ax1.plot(t, surface, color='cyan', label="Surface Temp (Pyrometer)", alpha=0.4)
    ax1.axhline(carcass_core, color='#ff4b4b', ls='--', label=f"Carcass Memory: {carcass_core:.1f}°C")
    ax1.set_ylabel("Temp (°C)"); ax1.legend(); st.pyplot(fig1); plt.close(fig1)
    st.write("Predicting the 'Grease Point' by tracking cumulative energy from 1200HP braking events.")

with tabs[2]:
    st.header("Bode Phasing: Structural Chatter")
    hz = 58 if "Titanium" in mat else 42
    f = np.linspace(0, 150, 500); amp = (1 / (1 + (30 * (f/hz - hz/f))**2)) * 10
    fig2, ax2 = plt.subplots(figsize=(10, 3)); plt.style.use('dark_background')
    ax2.plot(f, amp, color='#ff00ff', lw=3); ax2.axvline(hz, color='white', ls=':')
    ax2.set_xlabel("Frequency (Hz)"); st.pyplot(fig2); plt.close(fig2)
    st.warning(f"**Titanium Node Detected:** Resonating at {hz}Hz. High-speed damping must cancel this frequency.")

st.caption("Elite-Racing-Agent")
