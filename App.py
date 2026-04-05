# --- 3. THE UNIFIED PHYSICS & AI ENGINE ---
def run_sovereign_v16(hp, kg, rho, mat, wing_choice):
    # This function now uses 'wing_choice' to stay consistent
    x_mu, y_aero = np.meshgrid(np.linspace(1.0, 3.0, 100), np.linspace(0, 1500, 100))
    
    # AI Inference for the Optimal Point
    mu_target = 2.22 
    a_target = 950 if wing_choice == "Triple-Element" else 650
    
    rv = multivariate_normal([mu_target, a_target], [[0.08, 0], [0, 7500]])
    Z = rv.pdf(np.dstack((x_mu, y_aero))) * 1000
    idx = np.unravel_index(np.argmax(Z), Z.shape)
    o_mu, o_aero = x_mu[idx], y_aero[idx]
    
    vel = np.linspace(0, 350, 100)
    aoa_range = np.linspace(0, 25, 100)
    freq_range = np.linspace(0, 250, 200)
    time_range = np.linspace(0, 90, 100)
    
    return x_mu, y_aero, Z, o_mu, o_aero, vel, aoa_range, freq_range, time_range

# CORRECTED FUNCTION CALL: Passing 'wing_elements' from the sidebar into the function
XM, YM, ZM, OM, OA, V, AOA, F, T = run_sovereign_v16(hp, kg, rho_s, mat_upright, wing_elements)

# CORRECTED LINE 75 (Inside the RL Tab):
with tabs[1]:
    st.header("PPO Reinforcement Learning: Wing AoA")
    # Using 'wing_elements' here ensures the error disappears
    rew = norm.pdf(AOA, 13 if wing_elements == "Triple-Element" else 8, 3) * 100
    fig2, ax2 = plt.subplots(figsize=(10, 3)); plt.style.use('dark_background')
    ax2.plot(AOA, rew, color='#00ff9d', lw=3); ax2.set_xlabel("Angle of Attack (deg)"); ax2.set_ylabel("Neural Reward"); st.pyplot(fig2)
