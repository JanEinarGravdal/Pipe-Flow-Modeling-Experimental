# -*- coding: utf-8 -*-
"""
Streamlit App for MOC Non-Newtonian Pipe Flow Simulator (v3.3 - Interactive Batching)

Transforms the app into an interactive simulator allowing real-time
adjustment of inlet flow rate during the run via session state management.
Density changes apply on Reset/Start.
v3.1: Fixes NameError by re-inserting the calculate_initial_conditions function definition.
v3.2: Replaces deprecated st.experimental_rerun() with st.rerun().
v3.3: Implements batching of simulation steps before UI update to improve plot responsiveness.
"""

import streamlit as st
import numpy as np
import math
import pandas as pd
import plotly.graph_objects as go
import traceback
import time # Required for sleep

# --- Simulation Constants ---
# Adjust this number to balance responsiveness and smoothness
# Higher value = fewer UI updates = smoother but less "live" feeling
# Lower value = more UI updates = more "live" but potentially laggy plots
STEPS_PER_UPDATE = 15

# ==============================================================================
# --- Helper Functions (Unchanged from v3.2) ---
# ==============================================================================

def calculate_Re_MR(V, D, n, K, rho_fluid):
    """Calculates the Metzner-Reed Reynolds number."""
    V = float(V); D = float(D); n = float(n); K = float(K); rho_fluid = float(rho_fluid)
    if abs(V) <= 1e-9: return 0.0
    if D <= 0 or K <= 0 or rho_fluid <= 0: return 0.0
    if abs(n) < 1e-9: n = 1e-9
    shear_rate_factor = (3 * n + 1) / (4 * n)
    gamma_dot_eff = shear_rate_factor * (8 * abs(V) / D)
    if abs(n - 1.0) < 1e-6: mu_app = K
    elif gamma_dot_eff <= 1e-9: mu_app = K * (1e-9)**(n - 1) if n < 1 else float('inf')
    else: mu_app = K * (gamma_dot_eff)**(n - 1)
    if mu_app <= 1e-12 or mu_app == float('inf'): return 0.0
    if abs(n) < 1e-9: n_eff_kp = 1e-9
    else: n_eff_kp = n
    safe_shear_factor = max(abs(shear_rate_factor), 1e-12)
    Kp_factor = (safe_shear_factor)**n_eff_kp * 8**(n_eff_kp - 1)
    K_p = K * Kp_factor
    if K_p <= 1e-12: return 0.0
    try: Re_MR = rho_fluid * (abs(V)**(2.0 - n)) * (D**n) / K_p
    except (OverflowError, ValueError): Re_MR = float('inf')
    return Re_MR

def friction_factor_turbulent_straight(Re_MR, rel_roughness, n_fluid):
    """Calculates turbulent friction factor approximation."""
    Re = float(Re_MR); rel_roughness = float(rel_roughness)
    RE_TURBULENT_LIMIT_NN = 4000.0
    if Re <= RE_TURBULENT_LIMIT_NN: Re = RE_TURBULENT_LIMIT_NN + 1e-3
    if rel_roughness < 0: rel_roughness = 0
    term1_eps = rel_roughness / 3.7
    term2_re = 12.0 / Re if Re > 0 else 0
    psi1 = -2.0 * np.log10( term1_eps + term2_re ) if (term1_eps + term2_re) > 1e-12 else 0
    term3_re = 2.51 * psi1 / Re if Re > 0 else 0
    psi2 = -2.0 * np.log10( term1_eps + term3_re ) if (term1_eps + term3_re) > 1e-12 else 0
    term4_re = 2.51 * psi2 / Re if Re > 0 else 0
    psi3 = -2.0 * np.log10( term1_eps + term4_re ) if (term1_eps + term4_re) > 1e-12 else 0
    denom = (psi3 - 2.0 * psi2 + psi1)
    if abs(denom) < 1e-9 or psi1 <=0 or psi2 <=0: f_s_turb = 0.03
    else: f_s_turb = (psi1 - (psi2 - psi1)**2.0 / denom)**-2.0
    f_s_turb = max(f_s_turb, 0.008)
    return f_s_turb

def friction_factor_coiled(Re_MR, n_fluid, D_pipe, D_coil, roughness, rho_fluid, K_pl):
    """ Calculates friction factor for coiled pipe flow (Power Law). """
    RE_LAMINAR_LIMIT_NN = 2100.0
    RE_TURBULENT_LIMIT_NN = 4000.0
    Re_MR=float(Re_MR); n_fluid=float(n_fluid); D_pipe=float(D_pipe); D_coil=float(D_coil); roughness=float(roughness)
    f_s_lam = 64.0 / Re_MR if Re_MR > 1e-9 else float('inf')
    De = 0.0
    if D_coil > 1e-6 and D_pipe > 0:
        curvature_ratio = D_pipe / D_coil
        if curvature_ratio >= 0:
             try: De = Re_MR * np.sqrt(max(0, curvature_ratio))
             except ValueError: De = 0.0
    rel_roughness = roughness / D_pipe if D_pipe > 0 else 0
    if Re_MR < RE_LAMINAR_LIMIT_NN:
        regime = "Laminar"; f_c = f_s_lam
        if De > 1e-6 and n_fluid > 0:
             try:
                 De_base = max(De, 1e-9); exponent_val = 1.0 - 0.5 * n_fluid
                 De_term = De_base**exponent_val if De_base > 0 or exponent_val >= 0 else 0
                 correction_factor = 1.0 + 0.091 * (max(n_fluid, 1e-9)**0.4) * De_term
                 f_c = f_s_lam * correction_factor; f_c = max(f_c, f_s_lam)
             except (ValueError, OverflowError): f_c = f_s_lam
        f_c = max(0.0, f_c) if not np.isinf(f_c) else 0.0
    elif Re_MR > RE_TURBULENT_LIMIT_NN:
        regime = "Turbulent"
        f_s_turb = friction_factor_turbulent_straight(Re_MR, rel_roughness, n_fluid); f_c = f_s_turb
        if De > 1e-6:
             try:
                 De_base = max(De, 1e-9); correction_turb = De_base**0.05 if De_base > 0 else 1.0
                 f_c = f_s_turb * max(1.0, correction_turb)
             except (ValueError, OverflowError): f_c = f_s_turb
        f_c = max(f_c, 0.008)
    else:
        regime = "Transition"
        Re_lam_limit_calc = RE_LAMINAR_LIMIT_NN
        f_s_lam_limit = 64.0 / Re_lam_limit_calc if Re_lam_limit_calc > 1e-9 else float('inf')
        De_lam_limit = Re_lam_limit_calc * np.sqrt(max(0, D_pipe / D_coil)) if D_coil > 1e-6 else 0
        f_c_lam_limit = f_s_lam_limit
        if De_lam_limit > 1e-6 and n_fluid > 0:
            try:
                exponent_val = 1.0 - 0.5 * n_fluid; De_base = max(De_lam_limit, 1e-9)
                De_term = De_base**exponent_val if De_base > 0 or exponent_val >= 0 else 0
                correction_factor = 1.0 + 0.091 * (max(n_fluid, 1e-9)**0.4) * De_term
                f_c_lam_limit = f_s_lam_limit * correction_factor; f_c_lam_limit = max(f_c_lam_limit, f_s_lam_limit)
            except (ValueError, OverflowError): pass
        f_c_lam_limit = max(0.0, f_c_lam_limit) if not np.isinf(f_c_lam_limit) else 0.0
        Re_turb_limit_calc = RE_TURBULENT_LIMIT_NN
        f_s_turb_limit = friction_factor_turbulent_straight(Re_turb_limit_calc, rel_roughness, n_fluid)
        De_turb_limit = Re_turb_limit_calc * np.sqrt(max(0, D_pipe / D_coil)) if D_coil > 1e-6 else 0
        f_c_turb_limit = f_s_turb_limit
        if De_turb_limit > 1e-6:
             try:
                 De_base = max(De_turb_limit, 1e-9); correction_turb = De_base**0.05 if De_base > 0 else 1.0
                 f_c_turb_limit = f_s_turb_limit * max(1.0, correction_turb)
             except (ValueError, OverflowError): pass
        f_c_turb_limit = max(f_c_turb_limit, 0.008)
        if Re_MR <= Re_lam_limit_calc + 1e-6 : f_c = f_c_lam_limit
        elif Re_MR >= Re_turb_limit_calc - 1e-6: f_c = f_c_turb_limit
        elif abs(Re_turb_limit_calc - Re_lam_limit_calc) < 1e-6: f_c = (f_c_lam_limit + f_c_turb_limit) / 2.0
        elif f_c_turb_limit <= 1e-9 or f_c_lam_limit <= 1e-9:
             w = (Re_MR - Re_lam_limit_calc) / (Re_turb_limit_calc - Re_lam_limit_calc)
             f_c = f_c_lam_limit + w * (f_c_turb_limit - f_c_lam_limit)
        else:
             try:
                 log_Re_MR=np.log10(Re_MR); log_Re_lam=np.log10(Re_lam_limit_calc); log_Re_turb=np.log10(Re_turb_limit_calc)
                 log_f_lam=np.log10(f_c_lam_limit); log_f_turb=np.log10(f_c_turb_limit)
                 if abs(log_Re_turb - log_Re_lam) < 1e-9: f_c = f_c_lam_limit
                 else: log_f = log_f_lam + (log_f_turb - log_f_lam) * (log_Re_MR - log_Re_lam) / (log_Re_turb - log_Re_lam); f_c = 10**log_f
             except (ValueError, OverflowError): w = (Re_MR - Re_lam_limit_calc) / (Re_turb_limit_calc - Re_lam_limit_calc); f_c = f_c_lam_limit + w * (f_c_turb_limit - f_c_lam_limit)
        f_c = max(0.0, f_c)
    return f_c, De, regime

def linear_interpolate(y1, y2, x_interval, x_target_rel):
    """ Linear interpolation: y at x_target_rel within interval [0, x_interval] """
    if x_interval <= 1e-9: return y1
    interp_factor = max(0.0, min(1.0, x_target_rel / x_interval))
    return y1 + (y2 - y1) * interp_factor

def calculate_initial_conditions(Q0, P0_out, P_arr, V_arr, params):
    """ Calculates initial steady state conditions - MATCHING ORIGINAL SCRIPT LOGIC """
    status = []
    N = params['N']; dx = params['dx']; A = params['A']
    D = params['D']; n_pl = params['n_pl']; K_pl = params['K_pl']
    rho = params['rho']; D_c = params['D_c']; roughness = params['roughness']
    g = params['g']; theta_rad = params['theta_rad']
    status.append("info: Calculating initial conditions...")
    if A <= 0: raise ValueError("Pipe area must be positive.")
    V0 = Q0 / A
    V_arr[:] = V0
    Re0 = calculate_Re_MR(V0, D, n_pl, K_pl, rho)
    if Re0 == float('inf') or np.isnan(Re0): raise ValueError(f"Initial Reynolds number calculation failed (V0={V0}, D={D}, n={n_pl}, K={K_pl}, rho={rho})")
    f_c0, De0, regime0 = friction_factor_coiled(Re0, n_pl, D, D_c, roughness, rho, K_pl)
    status.append(f"info: Initial State - Re_MR={Re0:.1f}, De={De0:.1f}, Regime='{regime0}', f_c={f_c0:.4f}")
    if D <= 0: raise ValueError("Pipe diameter must be positive.")
    friction_term0_orig = f_c0 * V0 * abs(V0) / (2.0 * D) if D > 0 else 0
    gravity_term0_orig = g * math.sin(theta_rad)
    RHS_steady_0_orig = - friction_term0_orig - gravity_term0_orig
    P_arr[N] = P0_out
    for j in range(N - 1, -1, -1): P_arr[j] = P_arr[j+1] - dx * RHS_steady_0_orig
    status.append(f"info: Initial conditions calculated: V0={V0:.3f} m/s, P_in={P_arr[0]/1e5:.2f} bar (Using original IC method)")
    return status

# ==============================================================================
# --- Simulation State Initialization ---
# ==============================================================================

def initialize_simulation_state(params):
    """Initializes or resets the simulation state in st.session_state."""
    st.session_state.sim_status = "stopped"
    st.session_state.current_t = 0.0
    st.session_state.log_messages = ["info: Simulation state initialized."]
    N = params['N']; L = params['L']
    A = params.get('A', 0)
    if A <= 0:
        st.session_state.log_messages.append("error: Pipe Area (A) is zero during initialization.")
        st.session_state.sim_status = "error"; return
    initial_flow_lpm = params['q_in_config'].get('flow_min_lpm', 0.0)
    initial_Q0 = initial_flow_lpm * 0.001 / 60.0
    P0_out = params['P_out_val']
    st.session_state.run_params = params.copy()
    try:
        st.session_state.num_nodes = N + 1
        st.session_state.x_nodes = np.linspace(0, L, st.session_state.num_nodes)
        st.session_state.P = np.zeros(st.session_state.num_nodes)
        st.session_state.V = np.zeros(st.session_state.num_nodes)
        init_params_subset = {k: params[k] for k in ['N', 'dx', 'A', 'D', 'n_pl', 'K_pl', 'rho', 'D_c', 'roughness', 'g', 'theta_rad'] if k in params}
        init_status = calculate_initial_conditions(initial_Q0, P0_out, st.session_state.P, st.session_state.V, init_params_subset)
        st.session_state.log_messages.extend(init_status)
        st.session_state.time_history = [st.session_state.current_t]
        st.session_state.P_in_history = [st.session_state.P[0]]
        st.session_state.Q_in_history = [st.session_state.V[0] * A * 60000.]
        st.session_state.P_out_history = [st.session_state.P[N]]
        st.session_state.Q_out_history = [st.session_state.V[N] * A * 60000.]
        st.session_state.history_step_counter = 0
        st.session_state.history_store_interval = 10 # Example: Store every 10 steps
    except Exception as e:
        st.session_state.log_messages.append(f"error: Failed to initialize simulation state: {e}")
        st.session_state.sim_status = "error"

# ==============================================================================
# --- Single Step MOC Function ---
# ==============================================================================

def run_moc_one_step(P, V, t, dt, current_params, live_inlet_Q_m3s):
    """ Runs the MOC calculation for ONE time step dt. """
    status_messages_step = []
    N = current_params['N']; dx = current_params['dx']; rho = current_params['rho']
    a = current_params['a']; g = current_params['g']; theta_rad = current_params['theta_rad']
    D = current_params['D']; A = current_params['A']; n_pl = current_params['n_pl']
    K_pl = current_params['K_pl']; D_c = current_params['D_c']; roughness = current_params['roughness']
    P_out_val = current_params['P_out_val']
    P_new = np.zeros_like(P)
    V_new = np.zeros_like(V)
    try:
        # Boundary Conditions
        V_new[0] = live_inlet_Q_m3s / A if A > 0 else 0
        P_S = linear_interpolate(P[0], P[1], dx, a*dt)
        V_S = linear_interpolate(V[0], V[1], dx, a*dt)
        Re_S = calculate_Re_MR(V_S, D, n_pl, K_pl, rho)
        f_c_S, _, _ = friction_factor_coiled(Re_S, n_pl, D, D_c, roughness, rho, K_pl)
        FrictionTerm_S = f_c_S * V_S * abs(V_S) / (2.0 * D) if D > 0 else 0
        GravityTerm_BC = g * math.sin(theta_rad)
        P_new[0] = P_S + rho*a * (V_new[0] - V_S) + rho*a*(GravityTerm_BC + FrictionTerm_S)*dt
        P_new[N] = P_out_val
        x_R_rel_from_N_minus_1 = dx - a*dt
        P_R_BC = linear_interpolate(P[N-1], P[N], dx, x_R_rel_from_N_minus_1)
        V_R_BC = linear_interpolate(V[N-1], V[N], dx, x_R_rel_from_N_minus_1)
        Re_R_BC = calculate_Re_MR(V_R_BC, D, n_pl, K_pl, rho)
        f_c_R_BC, _, _ = friction_factor_coiled(Re_R_BC, n_pl, D, D_c, roughness, rho, K_pl)
        FrictionTerm_R_BC = f_c_R_BC * V_R_BC * abs(V_R_BC) / (2.0 * D) if D > 0 else 0
        rho_a_safe_BC = rho*a + 1e-12
        V_new[N] = V_R_BC - (1.0/rho_a_safe_BC)*(P_new[N] - P_R_BC) - (GravityTerm_BC + FrictionTerm_R_BC)*dt

        # Interior Nodes
        GravityTerm_Loop = g * math.sin(theta_rad)
        rho_a_safe_Loop = rho * a + 1e-12
        for j in range(1, N):
            x_R_rel_from_j_minus_1 = dx - a*dt
            P_R = linear_interpolate(P[j-1], P[j], dx, x_R_rel_from_j_minus_1)
            V_R = linear_interpolate(V[j-1], V[j], dx, x_R_rel_from_j_minus_1)
            Re_R = calculate_Re_MR(V_R, D, n_pl, K_pl, rho)
            f_c_R, _, _ = friction_factor_coiled(Re_R, n_pl, D, D_c, roughness, rho, K_pl)
            FrictionTerm_R = f_c_R * V_R * abs(V_R) / (2.0 * D) if D > 0 else 0
            x_S_rel_from_j = a*dt
            P_S = linear_interpolate(P[j], P[j+1], dx, x_S_rel_from_j)
            V_S = linear_interpolate(V[j], V[j+1], dx, x_S_rel_from_j)
            Re_S = calculate_Re_MR(V_S, D, n_pl, K_pl, rho)
            f_c_S, _, _ = friction_factor_coiled(Re_S, n_pl, D, D_c, roughness, rho, K_pl)
            FrictionTerm_S = f_c_S * V_S * abs(V_S) / (2.0 * D) if D > 0 else 0
            B_R = V_R - (GravityTerm_Loop + FrictionTerm_R)*dt
            B_S = V_S - (GravityTerm_Loop + FrictionTerm_S)*dt
            V_new[j] = 0.5 * ( B_R + B_S + (P_R - P_S)/rho_a_safe_Loop )
            P_new[j] = 0.5 * ( P_R + P_S + rho_a_safe_Loop*(B_R - B_S) )
            if np.isnan(P_new[j]) or np.isnan(V_new[j]): raise ValueError(f"NaN detected at node {j}, t={t+dt:.3f}.")
        return P_new, V_new, status_messages_step
    except Exception as e:
        status_messages_step.append(f"error: Error during MOC step at t={t:.3f}s: {e}")
        return P, V, status_messages_step # Return original state on error


# ==============================================================================
# --- Streamlit User Interface Code ---
# ==============================================================================

st.set_page_config(layout="wide", page_title="Interactive MOC Simulator")
st.title("🌊 Interactive MOC Non-Newtonian Pipe Flow Simulator (v3.3)")
st.markdown("Adjust Inlet Flow Rate live during simulation. Density changes apply on Reset/Start.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Simulation Setup")
    st.markdown("**Note:** Change these parameters and click 'Reset & Apply' before starting.")
    with st.expander("Pipe Geometry", expanded=False):
        L = st.number_input("Pipe Length (L, m)", value=2300.0, min_value=1.0, step=10.0, key="L_input")
        D = st.number_input("Pipe Diameter (D, m)", value=0.05, min_value=0.001, step=0.005, format="%.3f", key="D_input")
        D_c = st.number_input("Coil Diameter (Dc, m)", value=5.0, min_value=D*2 if D else 0.1, step=0.1, help="Outer diameter of the coil loop.", key="Dc_input")
        roughness = st.number_input("Pipe Roughness (ε, m)", value=0.000045, min_value=0.0, step=0.00001, format="%.6f", key="rough_input")
        angle_deg = st.slider("Pipe Angle (degrees from horizontal)", -90.0, 90.0, 0.0, 1.0, key="angle_input")
    with st.expander("Fluid Properties", expanded=True):
        rho_input = st.number_input("Fluid Density (ρ, kg/m³)", value=1000.0, min_value=100.0, step=10.0, key="rho_widget")
        K_pl = st.number_input("Consistency Index (K, Pa.sⁿ)", value=0.065, min_value=0.001, step=0.005, format="%.4f", key="Kpl_input")
        n_pl = st.number_input("Flow Behaviour Index (n)", value=0.9, min_value=0.1, max_value=2.0, step=0.05, format="%.2f", help="n<1: Shear-thinning, n=1: Newtonian, n>1: Shear-thickening", key="npl_input")
        K_bulk = st.number_input("Fluid Bulk Modulus (K_bulk, Pa)", value=1.0e9, min_value=1e6, step=0.1e9, format="%.2e", key="Kbulk_input")
    with st.expander("Pipe Properties"):
        E_pipe = st.number_input("Pipe Young's Modulus (E, Pa)", value=200e9, min_value=1e9, step=10e9, format="%.2e", key="Epipe_input")
        e_pipe = st.number_input("Pipe Wall Thickness (e, m)", value=0.005, min_value=0.001, step=0.001, format="%.4f", key="epipe_input")
        c1_constraint = st.number_input("Pipe Constraint Factor (c1)", value=1.0, min_value=0.1, max_value=1.5, step=0.1, help="Accounts for pipe anchoring effects on wave speed.", key="c1_input")
    with st.expander("Simulation Control", expanded=False):
        N = st.slider("Number of Spatial Segments (N)", 10, 500, 50, 5, help="More segments increase accuracy but slow down simulation.", key="N_input")
        target_Cr = st.slider("Target Courant Number (Cr)", 0.1, 1.0, 0.95, 0.05, help="Stability requires Cr <= 1. dt is calculated based on this.", key="Cr_input")
        t_final = st.number_input("Max Simulation Time (s)", value=100.0, min_value=1.0, step=10.0, key="tfinal_input")
    with st.expander("Initial/Outlet Conditions"):
        P_out_val = st.number_input("Outlet Pressure (P_out, Pa)", value=1.0e5, step=0.1e5, format="%.1e", key="Pout_input")
        initial_flow_lpm = st.number_input("Initial Flow Rate (L/min, for t=0)", value=0.0, min_value=0.0, key="init_flow_input")

    st.sidebar.markdown("---")
    st.header("🚀 Live Controls")
    live_flow_rate_lpm = st.slider("LIVE Inlet Flow Rate (L/min)", min_value=0.0, max_value=1000.0, value=initial_flow_lpm, step=10.0, key="live_flow_slider")
    col1, col2, col3 = st.columns(3)
    with col1: start_button = st.button("▶️ Start", key="start_btn")
    with col2: pause_button = st.button("⏸️ Pause", key="pause_btn")
    with col3: reset_button = st.button("🔄 Reset & Apply", key="reset_btn")

# --- Parameter Assemby and State Initialization ---
dx = L / N if N > 0 else 0
A = math.pi * (D/2)**2 if D > 0 else 0
g = 9.81
theta_rad = math.radians(angle_deg)
dt_to_use = 0.05
a_est = 1000.0
try:
    if rho_input > 0 and K_bulk > 0 and E_pipe > 0 and e_pipe > 0 and D > 0:
        pipe_term_est = c1_constraint * K_bulk * D / (E_pipe * e_pipe)
        a_squared_est = (K_bulk / rho_input) / (1 + pipe_term_est)
        if a_squared_est > 0: a_est = math.sqrt(a_squared_est)
    if a_est > 0 and dx > 0:
         dt_to_use = target_Cr * dx / a_est
         if dt_to_use <=0: dt_to_use = 0.05
except Exception: pass

current_input_params = {
    'L': L, 'D': D, 'D_c': D_c, 'roughness': roughness, 'angle_deg': angle_deg,
    'rho': rho_input,
    'K_pl': K_pl, 'n_pl': n_pl,
    'K_bulk': K_bulk, 'E_pipe': E_pipe, 'e_pipe': e_pipe, 'c1_constraint': c1_constraint,
    'N': N, 'dt': dt_to_use, 't_final': t_final, 'Cr_target': target_Cr,
    'P_out_val': P_out_val,
    'q_in_config': {'flow_min_lpm': initial_flow_lpm},
    'dx': dx, 'A': A, 'g': g, 'theta_rad': theta_rad, 'a': a_est, 'Cr': target_Cr
}

if 'sim_status' not in st.session_state or reset_button:
    initialize_simulation_state(current_input_params)
    if reset_button:
        st.rerun() # Use st.rerun()

# --- Handle Button Clicks ---
if start_button:
    if st.session_state.sim_status in ["stopped", "paused"]:
        if st.session_state.current_t == 0.0 or st.session_state.sim_status == "stopped":
             st.session_state.log_messages.append("info: Starting simulation from t=0...")
             initialize_simulation_state(current_input_params)
             if st.session_state.sim_status == "error":
                 st.error("Initialization failed. Cannot start simulation. Check logs.")
             else:
                  st.session_state.sim_status = "running"
                  st.rerun() # Use st.rerun()
        else: # Resuming from pause
             st.session_state.log_messages.append("info: Resuming simulation...")
             st.session_state.sim_status = "running"
             st.rerun() # Use st.rerun()
elif pause_button:
    if st.session_state.sim_status == "running":
        st.session_state.log_messages.append("info: Simulation paused.")
        st.session_state.sim_status = "paused"
        st.rerun() # Use st.rerun()

# ==========================================================================
# == REVISED Main Simulation Step Execution Loop (Using Batching) =========
# ==========================================================================
if st.session_state.sim_status == "running":
    steps_taken_this_batch = 0
    # Inner loop to run multiple steps before updating UI
    while steps_taken_this_batch < STEPS_PER_UPDATE and st.session_state.current_t < st.session_state.run_params['t_final']:
        # Get live input value (only needs reading once per batch, but reading here is fine)
        live_Q_lpm = live_flow_rate_lpm
        live_Q_m3s = live_Q_lpm * 0.001 / 60.0

        # Get current state from session_state
        P_curr = st.session_state.P
        V_curr = st.session_state.V
        t_curr = st.session_state.current_t
        dt_run = st.session_state.run_params['dt']

        # Run one step
        P_next, V_next, step_msgs = run_moc_one_step(
            P_curr, V_curr, t_curr, dt_run, st.session_state.run_params, live_Q_m3s
        )
        st.session_state.log_messages.extend(step_msgs) # Append logs immediately

        # Check for errors from the step
        if any("error:" in msg for msg in step_msgs):
            st.session_state.sim_status = "error"
            st.error("Error occurred during simulation step. See log.")
            # Break the inner batch loop on error
            break
        else:
            # Update state IN SESSION STATE for the next inner step or final update
            st.session_state.P = P_next
            st.session_state.V = V_next
            st.session_state.current_t += dt_run
            st.session_state.history_step_counter += 1
            steps_taken_this_batch += 1

            # Store history (sampled based on overall counter)
            if st.session_state.history_step_counter % st.session_state.history_store_interval == 0:
                 A_hist = st.session_state.run_params.get('A', 0)
                 if A_hist > 0:
                     st.session_state.time_history.append(st.session_state.current_t)
                     st.session_state.P_in_history.append(P_next[0])
                     st.session_state.Q_in_history.append(V_next[0] * A_hist * 60000.)
                     st.session_state.P_out_history.append(P_next[st.session_state.run_params['N']])
                     st.session_state.Q_out_history.append(V_next[st.session_state.run_params['N']] * A_hist * 60000.)

    # --- After the inner loop (batch complete, error, or t_final reached) ---

    # Check if simulation finished or errored during the batch
    if st.session_state.sim_status == "error" or st.session_state.current_t >= st.session_state.run_params['t_final']:
         # Handle completion
         if st.session_state.current_t >= st.session_state.run_params['t_final'] and st.session_state.sim_status != "error":
            st.session_state.log_messages.append("info: Simulation reached t_final.")
            st.session_state.sim_status = "stopped"
            st.success(f"Simulation completed at t = {st.session_state.current_t:.2f} s.")
         # Trigger a final rerun to update UI to stopped/error state and show final plots
         st.rerun() # Use st.rerun()
    elif st.session_state.sim_status == "running":
         # Batch finished, still running -> trigger UI update and next batch
         # No explicit sleep needed here usually, the rerun handles the pause for rendering
         st.rerun() # Use st.rerun()
    # If paused externally during the batch, the outer `if st.session_state.sim_status == "running":`
    # check will fail on the next rerun, effectively stopping the loop.

# ==========================================================================
# == END REVISED Main Simulation Step Execution Loop =======================
# ==========================================================================


# --- Display Status and Results ---
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Simulation Status")
sim_status_indicator = st.session_state.get('sim_status', 'stopped')
status_time = st.session_state.get('current_t', 0.0)
if sim_status_indicator == "running":
    st.sidebar.success(f"Running... t = {status_time:.2f} s")
elif sim_status_indicator == "paused":
    st.sidebar.warning(f"Paused at t = {status_time:.2f} s")
elif sim_status_indicator == "stopped":
    st.sidebar.info(f"Stopped at t = {status_time:.2f} s")
elif sim_status_indicator == "error":
    st.sidebar.error("Error State!")

st.subheader("📜 Simulation Log")
with st.expander("Show Log Details", expanded=False):
    log_display_limit = 20
    for msg in reversed(st.session_state.get('log_messages', [])[-log_display_limit:]):
        if msg.startswith("info:") or msg.startswith("progress:"): st.info(msg[msg.find(":")+1:].strip())
        elif msg.startswith("warning:"): st.warning(msg[msg.find(":")+1:].strip())
        elif msg.startswith("error:"): st.error(msg[msg.find(":")+1:].strip())
        else: st.write(msg)

st.subheader("📈 Live Results")
results_placeholder = st.empty()
with results_placeholder.container():
    if 'P' in st.session_state and 'V' in st.session_state and 'x_nodes' in st.session_state and 'run_params' in st.session_state:
        P_disp = st.session_state.P
        V_disp = st.session_state.V
        x_nodes_disp = st.session_state.x_nodes
        N_disp = st.session_state.run_params.get('N', 1)
        A_disp = st.session_state.run_params.get('A', 0)
        Q_disp_lpm = V_disp * A_disp * 60000. if A_disp > 0 else np.zeros_like(V_disp)

        tab1, tab2, tab3 = st.tabs(["Live Profiles", "Time Histories", "Key Metrics"])

        with tab1:
            st.markdown(f"#### Profiles at t = {st.session_state.current_t:.2f} s")
            fig_p_profile = go.Figure()
            fig_p_profile.add_trace(go.Scatter(x=x_nodes_disp, y=P_disp / 1e5, name='Pressure', mode='lines'))
            fig_p_profile.update_layout(title="Live Pressure Profile", xaxis_title="Pipe Length (m)", yaxis_title="Pressure (bar)", uirevision='live_p_profile') # Add uirevision
            st.plotly_chart(fig_p_profile, use_container_width=True)
            fig_q_profile = go.Figure()
            fig_q_profile.add_trace(go.Scatter(x=x_nodes_disp, y=Q_disp_lpm, name='Flow Rate', mode='lines'))
            fig_q_profile.update_layout(title="Live Flow Rate Profile", xaxis_title="Pipe Length (m)", yaxis_title="Flow Rate (L/min)", uirevision='live_q_profile') # Add uirevision
            st.plotly_chart(fig_q_profile, use_container_width=True)

        with tab2:
            st.markdown("#### History (Sampled)")
            if 'time_history' in st.session_state and len(st.session_state.time_history) > 1:
                fig_p_hist = go.Figure()
                fig_p_hist.add_trace(go.Scatter(x=st.session_state.time_history, y=np.array(st.session_state.P_in_history) / 1e5, name='Inlet P (bar)', mode='lines'))
                fig_p_hist.add_trace(go.Scatter(x=st.session_state.time_history, y=np.array(st.session_state.P_out_history) / 1e5, name='Outlet P (bar)', mode='lines', line=dict(dash='dash')))
                fig_p_hist.update_layout(xaxis_title="Time (s)", yaxis_title="Pressure (bar)", uirevision='p_history') # Add uirevision
                st.plotly_chart(fig_p_hist, use_container_width=True)
                fig_q_hist = go.Figure()
                fig_q_hist.add_trace(go.Scatter(x=st.session_state.time_history, y=st.session_state.Q_in_history, name='Inlet Q (L/min)', mode='lines'))
                fig_q_hist.add_trace(go.Scatter(x=st.session_state.time_history, y=st.session_state.Q_out_history, name='Outlet Q (L/min)', mode='lines', line=dict(dash='dash')))
                fig_q_hist.update_layout(xaxis_title="Time (s)", yaxis_title="Flow Rate (L/min)", uirevision='q_history') # Add uirevision
                st.plotly_chart(fig_q_hist, use_container_width=True)
            else: st.info("Run simulation to generate history plots.")

        with tab3:
             st.markdown("#### Live Metrics")
             col1, col2 = st.columns(2)
             with col1:
                 st.metric("Current Inlet Pressure (bar)", f"{P_disp[0]/1e5:.2f}")
                 st.metric("Current Outlet Flow (L/min)", f"{Q_disp_lpm[N_disp]:.1f}")
             with col2:
                 st.metric("Current Pressure Drop (bar)", f"{(P_disp[0] - P_disp[N_disp])/1e5:.2f}")
                 st.metric("Current Inlet Flow (L/min)", f"{Q_disp_lpm[0]:.1f}")
    else: st.info("Simulation state not initialized. Click 'Reset & Apply' or 'Start'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Interactive MOC Simulator (v3.3)")