# -*- coding: utf-8 -*-
"""
Streamlit App for MOC Non-Newtonian Pipe Flow Simulation (Revised v2)

Combines the MOC simulation logic with a Streamlit user interface.
Revision 2 incorporates changes to initial conditions and interior node
calculations to strictly match the user's original standalone script logic.
"""

import streamlit as st
import numpy as np
import math
import pandas as pd # Used for easier data handling if needed
import plotly.graph_objects as go
import traceback # For capturing detailed errors if needed

# --- Helper Functions (Adapted from original script) ---

def calculate_Re_MR(V, D, n, K, rho_fluid):
    """Calculates the Metzner-Reed Reynolds number."""
    V = float(V); D = float(D); n = float(n); K = float(K); rho_fluid = float(rho_fluid)
    if abs(V) <= 1e-9: return 0.0
    if D <= 0 or K <= 0 or rho_fluid <= 0: return 0.0 # Added rho check
    if abs(n) < 1e-9: n = 1e-9 # Avoid division by zero

    shear_rate_factor = (3 * n + 1) / (4 * n)
    gamma_dot_eff = shear_rate_factor * (8 * abs(V) / D)

    # Avoid issues with n=1 (Newtonian limit) or gamma_dot_eff=0
    if abs(n - 1.0) < 1e-6:
        mu_app = K # For Newtonian, K is viscosity
    elif gamma_dot_eff <= 1e-9:
         mu_app = K * (1e-9)**(n - 1) if n < 1 else float('inf') # Example limiting behavior
    else:
        mu_app = K * (gamma_dot_eff)**(n - 1)

    if mu_app <= 1e-12 or mu_app == float('inf'): return 0.0 # Invalid apparent viscosity

    if abs(n) < 1e-9: n_eff_kp = 1e-9
    else: n_eff_kp = n

    safe_shear_factor = max(abs(shear_rate_factor), 1e-12)

    Kp_factor = (safe_shear_factor)**n_eff_kp * 8**(n_eff_kp - 1)
    K_p = K * Kp_factor

    if K_p <= 1e-12: return 0.0 # Avoid division by zero

    try:
      Re_MR = rho_fluid * (abs(V)**(2.0 - n)) * (D**n) / K_p
    except (OverflowError, ValueError):
        Re_MR = float('inf') # Handle potential math errors

    return Re_MR


def friction_factor_turbulent_straight(Re_MR, rel_roughness, n_fluid):
    """Calculates turbulent friction factor for power-law fluids in straight pipes.
       Uses an approximation (e.g., Serghides or similar iterative solution for Colebrook-White adapted for PL).
       Note: Accurate PL turbulent friction is complex. Using standard Colebrook-White approx. here.
    """
    Re = float(Re_MR); rel_roughness = float(rel_roughness)
    RE_TURBULENT_LIMIT_NN = 4000.0 # Placeholder

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
    if abs(denom) < 1e-9 or psi1 <=0 or psi2 <=0:
        f_s_turb = 0.03 # Fallback friction factor
    else:
        f_s_turb = (psi1 - (psi2 - psi1)**2.0 / denom)**-2.0

    f_s_turb = max(f_s_turb, 0.008)
    return f_s_turb

def friction_factor_coiled(Re_MR, n_fluid, D_pipe, D_coil, roughness, rho_fluid, K_pl):
    """ Calculates friction factor for coiled pipe flow (Power Law). """
    RE_LAMINAR_LIMIT_NN = 2100.0 # Placeholder
    RE_TURBULENT_LIMIT_NN = 4000.0 # Placeholder

    Re_MR = float(Re_MR); n_fluid = float(n_fluid); D_pipe = float(D_pipe); D_coil = float(D_coil); roughness = float(roughness)

    f_s_lam = 64.0 / Re_MR if Re_MR > 1e-9 else float('inf')

    De = 0.0
    if D_coil > 1e-6 and D_pipe > 0:
        curvature_ratio = D_pipe / D_coil
        if curvature_ratio >= 0:
             try:
                De = Re_MR * np.sqrt(max(0, curvature_ratio))
             except ValueError:
                De = 0.0

    rel_roughness = roughness / D_pipe if D_pipe > 0 else 0

    if Re_MR < RE_LAMINAR_LIMIT_NN:
        regime = "Laminar"
        f_c = f_s_lam
        if De > 1e-6 and n_fluid > 0:
             try:
                 De_base = max(De, 1e-9)
                 exponent_val = 1.0 - 0.5 * n_fluid
                 De_term = De_base**exponent_val if De_base > 0 or exponent_val >= 0 else 0
                 correction_factor = 1.0 + 0.091 * (max(n_fluid, 1e-9)**0.4) * De_term
                 f_c = f_s_lam * correction_factor
                 f_c = max(f_c, f_s_lam)
             except (ValueError, OverflowError):
                 f_c = f_s_lam
        f_c = max(0.0, f_c) if not np.isinf(f_c) else 0.0

    elif Re_MR > RE_TURBULENT_LIMIT_NN:
        regime = "Turbulent"
        f_s_turb = friction_factor_turbulent_straight(Re_MR, rel_roughness, n_fluid)
        f_c = f_s_turb
        if De > 1e-6:
             try:
                 De_base = max(De, 1e-9)
                 correction_turb = De_base**0.05 if De_base > 0 else 1.0
                 f_c = f_s_turb * max(1.0, correction_turb)
             except (ValueError, OverflowError):
                 f_c = f_s_turb
        f_c = max(f_c, 0.008)

    else: # Transition Regime
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
                 De_base = max(De_turb_limit, 1e-9)
                 correction_turb = De_base**0.05 if De_base > 0 else 1.0
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
                 log_Re_MR = np.log10(Re_MR); log_Re_lam = np.log10(Re_lam_limit_calc); log_Re_turb = np.log10(Re_turb_limit_calc)
                 log_f_lam = np.log10(f_c_lam_limit); log_f_turb = np.log10(f_c_turb_limit)
                 if abs(log_Re_turb - log_Re_lam) < 1e-9: f_c = f_c_lam_limit
                 else:
                     log_f = log_f_lam + (log_f_turb - log_f_lam) * (log_Re_MR - log_Re_lam) / (log_Re_turb - log_Re_lam)
                     f_c = 10**log_f
             except (ValueError, OverflowError):
                  w = (Re_MR - Re_lam_limit_calc) / (Re_turb_limit_calc - Re_lam_limit_calc)
                  f_c = f_c_lam_limit + w * (f_c_turb_limit - f_c_lam_limit)
        f_c = max(0.0, f_c)

    return f_c, De, regime

# ==============================================================================
# == REVISED Initial Conditions Function (Matches Original Script Logic) =====
# ==============================================================================
def calculate_initial_conditions(Q0, P0_out, P_arr, V_arr, params):
    """ Calculates initial steady state conditions - MATCHING ORIGINAL SCRIPT LOGIC """
    status = []
    # Unpack necessary params
    N = params['N']; dx = params['dx']; A = params['A']
    D = params['D']; n_pl = params['n_pl']; K_pl = params['K_pl']
    rho = params['rho']; D_c = params['D_c']; roughness = params['roughness']
    g = params['g']; theta_rad = params['theta_rad']

    status.append("info: Calculating initial conditions...")
    if A <= 0: raise ValueError("Pipe area must be positive.")
    V0 = Q0 / A
    V_arr[:] = V0

    Re0 = calculate_Re_MR(V0, D, n_pl, K_pl, rho)
    if Re0 == float('inf') or np.isnan(Re0):
        raise ValueError(f"Initial Reynolds number calculation failed (V0={V0}, D={D}, n={n_pl}, K={K_pl}, rho={rho})")

    # Note: friction_factor_coiled requires rho and K_pl now based on internal Re calc dependency
    f_c0, De0, regime0 = friction_factor_coiled(Re0, n_pl, D, D_c, roughness, rho, K_pl)
    status.append(f"info: Initial State - Re_MR={Re0:.1f}, De={De0:.1f}, Regime='{regime0}', f_c={f_c0:.4f}")

    if D <= 0: raise ValueError("Pipe diameter must be positive.")

    # --- MATCHING ORIGINAL RHS calculation and usage ---
    friction_term0_orig = f_c0 * V0 * abs(V0) / (2.0 * D) if D > 0 else 0 # As per original friction term calculation
    gravity_term0_orig = g * math.sin(theta_rad) # As per original gravity term calculation
    # This term (RHS_steady_0_orig) was used in the original script's integration step
    RHS_steady_0_orig = - friction_term0_orig - gravity_term0_orig

    P_arr[N] = P0_out # Pressure at outlet node
    for j in range(N - 1, -1, -1): # Integrate backwards from outlet
        # Original usage: P_j = P_{j+1} - dx * RHS_steady_0_orig
        # Replicating this exact calculation step:
        P_arr[j] = P_arr[j+1] - dx * RHS_steady_0_orig
    # --- END MATCHING ORIGINAL ---

    status.append(f"info: Initial conditions calculated: V0={V0:.3f} m/s, P_in={P_arr[0]/1e5:.2f} bar (Using original IC method)")
    return status
# ==============================================================================
# == END REVISED Initial Conditions Function ==================================
# ==============================================================================


def linear_interpolate(y1, y2, x_interval, x_target_rel):
    """ Linear interpolation: y at x_target_rel within interval [0, x_interval] """
    if x_interval <= 1e-9: return y1
    interp_factor = max(0.0, min(1.0, x_target_rel / x_interval))
    return y1 + (y2 - y1) * interp_factor

def Q_in_func(t, params):
    """ Function defining Inlet Flow Rate (m^3/s) at time t using parameters """
    t_ramp_start = params.get('t_ramp_start', 5.0)
    t_ramp_mid = params.get('t_ramp_mid', 35.0)
    t_ramp_end = params.get('t_ramp_end', 60.0)
    t_jump = params.get('t_jump', 80.0)
    flow_min_lpm = params.get('flow_min_lpm', 0.0)
    flow_mid_lpm = params.get('flow_mid_lpm', 400.0)
    flow_max_lpm = params.get('flow_max_lpm', 600.0)
    flow_jump_lpm = params.get('flow_jump_lpm', 800.0)

    if t < t_ramp_start: flow_lpm = flow_min_lpm
    elif t <= t_ramp_mid:
        mid_time = (t_ramp_start + t_ramp_mid) / 2.0
        width = (t_ramp_mid - t_ramp_start) / 2.0
        ramp_factor = 0.5 * (1 + np.tanh(3.0 * (t - mid_time) / max(1e-6, width)))
        flow_lpm = flow_min_lpm + (flow_mid_lpm - flow_min_lpm) * ramp_factor
    elif t > t_ramp_mid and t <= t_ramp_end:
        mid_time = (t_ramp_mid + t_ramp_end) / 2.0
        width = (t_ramp_end - t_ramp_mid) / 2.0
        ramp_factor = 0.5 * (1 + np.tanh(3.0 * (t - mid_time) / max(1e-6, width)))
        flow_lpm = flow_mid_lpm + (flow_max_lpm - flow_mid_lpm) * ramp_factor
    elif t > t_ramp_end and t <= t_jump:
        flow_lpm = flow_max_lpm
    else: # t > t_jump
        flow_lpm = flow_jump_lpm

    flow_m3s = flow_lpm * 0.001 / 60.0
    return max(0, flow_m3s)


# --- Main MOC Simulation Function ---
@st.cache_data
def run_moc_simulation(_params):
    """
    Runs the MOC simulation for non-Newtonian flow in coiled pipes. (Revised v2)

    Args:
        _params (dict): Dictionary containing all input parameters.

    Returns:
        tuple: (time_history, P_history, V_history, x_nodes, status_messages)
    """
    params = _params.copy()
    status_messages = []
    all_results = ([], [], [], [], status_messages)

    try:
        L = params['L']; D = params['D']; D_c = params['D_c']; roughness = params['roughness']
        angle_deg = params['angle_deg']; rho = params['rho']; K_pl = params['K_pl']; n_pl = params['n_pl']
        K_bulk = params['K_bulk']; E_pipe = params['E_pipe']; e_pipe = params['e_pipe']
        c1_constraint = params['c1_constraint']; N = params['N']; dt = params['dt']
        t_final = params['t_final']; P_out_val = params['P_out_val']
        q_in_config = params['q_in_config']

        if L <=0 or D <=0 or rho <=0 or K_pl <=0 or n_pl <= 0 or K_bulk <=0 or E_pipe <=0 or e_pipe <=0 or N <=0 or dt <=0 or t_final <=0:
             raise ValueError("Input parameters (L, D, rho, K_pl, n_pl, K_bulk, E_pipe, e_pipe, N, dt, t_final) must be positive.")

        dx = L / N
        params['dx'] = dx
        A = math.pi * (D/2)**2
        params['A'] = A
        g = 9.81
        params['g'] = g
        theta_rad = math.radians(angle_deg)
        params['theta_rad'] = theta_rad
        num_nodes = N + 1
        x_nodes = np.linspace(0, L, num_nodes)

        if rho <= 0 or E_pipe <= 0 or e_pipe <= 0 or D <= 0 or K_bulk <= 0:
             raise ValueError("Density, Young's Modulus, pipe thickness, diameter, and K_bulk must be positive for wave speed calc.")
        pipe_term = c1_constraint * K_bulk * D / (E_pipe * e_pipe) if E_pipe > 0 and e_pipe > 0 else 0
        wave_speed_denom = rho * (1 + pipe_term)
        if wave_speed_denom <= 0:
             raise ValueError("Invalid properties leading to non-positive wave speed denominator.")
        a_squared = (K_bulk / rho) / (1 + pipe_term)
        if a_squared <= 0:
             raise ValueError("Invalid properties leading to non-positive a^2.")
        a = math.sqrt(a_squared)
        params['a'] = a
        status_messages.append(f"info: Calculated Wave Speed a = {a:.2f} m/s")

        Cr = a * dt / dx
        params['Cr'] = Cr
        status_messages.append(f"info: Courant Number Cr = {Cr:.3f}")
        if Cr > 1.001:
            status_messages.append(f"warning: Courant condition violated (Cr = {Cr:.3f} > 1). Results may be inaccurate or unstable.")
        if Cr <= 0:
            raise ValueError(f"Courant number Cr={Cr:.3f} must be positive.")

        num_time_steps = int(t_final / dt) + 1
        if num_time_steps <= 1: raise ValueError("t_final must be greater than dt.")
        time_steps = np.linspace(0, t_final, num_time_steps)
        status_messages.append(f"info: Using N={N} segments (dx={dx:.2f}m), dt={dt:.4f}s ({num_time_steps} steps)")

        P = np.zeros(num_nodes)
        V = np.zeros(num_nodes)
        P_new = np.zeros(num_nodes)
        V_new = np.zeros(num_nodes)

        t_start = time_steps[0]
        Q0 = Q_in_func(t_start, q_in_config)
        P0_out = P_out_val
        # Call the REVISED initial conditions function
        init_status = calculate_initial_conditions(Q0, P0_out, P, V, params)
        status_messages.extend(init_status)

        time_history = [t_start]
        P_history = [P.copy()]
        V_history = [V.copy()]

        status_messages.append("info: --- Starting MOC Time Loop ---")

        for i in range(num_time_steps - 1):
            t_current = time_steps[i]
            t_next = time_steps[i+1]

            # --- Boundary Conditions (Checked against original - appear consistent) ---
            try:
                V_new[0] = Q_in_func(t_next, q_in_config) / A
                P_S = linear_interpolate(P[0], P[1], dx, a*dt)
                V_S = linear_interpolate(V[0], V[1], dx, a*dt)
                Re_S = calculate_Re_MR(V_S, D, n_pl, K_pl, rho)
                f_c_S, _, _ = friction_factor_coiled(Re_S, n_pl, D, D_c, roughness, rho, K_pl)
                FrictionTerm_S = f_c_S * V_S * abs(V_S) / (2.0 * D) if D > 0 else 0
                GravityTerm_BC = g * math.sin(theta_rad) # Renamed to avoid clash with loop var

                P_new[0] = P_S + rho*a * (V_new[0] - V_S) + rho*a*(GravityTerm_BC + FrictionTerm_S)*dt

                P_new[N] = P_out_val
                x_R_rel_from_N_minus_1 = dx - a*dt
                P_R_BC = linear_interpolate(P[N-1], P[N], dx, x_R_rel_from_N_minus_1) # Renamed P_R
                V_R_BC = linear_interpolate(V[N-1], V[N], dx, x_R_rel_from_N_minus_1) # Renamed V_R
                Re_R_BC = calculate_Re_MR(V_R_BC, D, n_pl, K_pl, rho)
                f_c_R_BC, _, _ = friction_factor_coiled(Re_R_BC, n_pl, D, D_c, roughness, rho, K_pl)
                FrictionTerm_R_BC = f_c_R_BC * V_R_BC * abs(V_R_BC) / (2.0 * D) if D > 0 else 0

                rho_a_safe_BC = rho*a + 1e-12
                V_new[N] = V_R_BC - (1.0/rho_a_safe_BC)*(P_new[N] - P_R_BC) - (GravityTerm_BC + FrictionTerm_R_BC)*dt

            except Exception as e:
                err_msg = f"error: ERROR calculating boundary conditions at t={t_next:.3f}s: {e}"
                status_messages.append(err_msg)
                return all_results

            # --- Interior Nodes (j=1 to N-1) ---
            GravityTerm_Loop = g * math.sin(theta_rad) # Use different name
            rho_a_safe_Loop = rho * a + 1e-12

            # ==========================================================================
            # == REVISED Interior Node Calculation (Matches Original B_R/B_S Logic) ==
            # ==========================================================================
            for j in range(1, N):
                try:
                    # Point R (along C+ from P(j, i+1)) at time i
                    x_R_rel_from_j_minus_1 = dx - a*dt
                    P_R = linear_interpolate(P[j-1], P[j], dx, x_R_rel_from_j_minus_1)
                    V_R = linear_interpolate(V[j-1], V[j], dx, x_R_rel_from_j_minus_1)
                    Re_R = calculate_Re_MR(V_R, D, n_pl, K_pl, rho)
                    f_c_R, _, _ = friction_factor_coiled(Re_R, n_pl, D, D_c, roughness, rho, K_pl)
                    FrictionTerm_R = f_c_R * V_R * abs(V_R) / (2.0 * D) if D > 0 else 0

                    # Point S (along C- from P(j, i+1)) at time i
                    x_S_rel_from_j = a*dt
                    P_S = linear_interpolate(P[j], P[j+1], dx, x_S_rel_from_j)
                    V_S = linear_interpolate(V[j], V[j+1], dx, x_S_rel_from_j)
                    Re_S = calculate_Re_MR(V_S, D, n_pl, K_pl, rho)
                    f_c_S, _, _ = friction_factor_coiled(Re_S, n_pl, D, D_c, roughness, rho, K_pl)
                    FrictionTerm_S = f_c_S * V_S * abs(V_S) / (2.0 * D) if D > 0 else 0

                    # --- Solve the 2x2 system using original B_R, B_S FORMULATION ---
                    B_R = V_R - (GravityTerm_Loop + FrictionTerm_R)*dt
                    # Matching original script implementation (likely '-' sign for B_S friction/gravity term)
                    B_S = V_S - (GravityTerm_Loop + FrictionTerm_S)*dt

                    # Calculate V_new[j] and P_new[j] using original B_R, B_S formulas
                    V_new[j] = 0.5 * ( B_R + B_S + (P_R - P_S)/rho_a_safe_Loop )
                    P_new[j] = 0.5 * ( P_R + P_S + rho_a_safe_Loop*(B_R - B_S) )
                    # --- END MATCHING ORIGINAL FORMULATION ---

                    if np.isnan(P_new[j]) or np.isnan(V_new[j]):
                        raise ValueError(f"NaN detected at node {j}, t={t_next:.3f}. Check inputs/stability.")

                except Exception as e:
                    err_msg = f"error: ERROR calculating interior node j={j} at t={t_next:.3f}s: {e}"
                    status_messages.append(err_msg)
                    return all_results # Stop simulation on node failure
            # ==========================================================================
            # == END REVISED Interior Node Calculation ================================
            # ==========================================================================


            # --- Update State for next iteration ---
            P[:] = P_new
            V[:] = V_new

            # --- Store Results ---
            time_history.append(t_next)
            P_history.append(P.copy())
            V_history.append(V.copy())

            if (i + 1) % (max(1, num_time_steps // 20)) == 0:
               status_messages.append(f"progress: t={t_next:.2f}s / {t_final:.2f}s")

        status_messages.append("info: Simulation Finished Successfully.")
        P_history = np.array(P_history)
        V_history = np.array(V_history)

        return time_history, P_history, V_history, x_nodes, status_messages

    except ValueError as ve:
        status_messages.append(f"error: Input Validation Error: {ve}")
        return all_results
    except Exception as e:
        status_messages.append(f"error: Critical failure during simulation setup or run: {e}")
        # status_messages.append(f"debug: {traceback.format_exc()}")
        return all_results

# ==============================================================================
# --- Streamlit User Interface Code (Mostly Unchanged) ---
# ==============================================================================

st.set_page_config(layout="wide", page_title="MOC Pipe Flow Simulator")
st.title("🌊 MOC Non-Newtonian Pipe Flow Simulator (Rev 2)")
st.markdown("Simulation of transient flow in a coiled pipe using the Method of Characteristics.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Simulation Inputs")

    with st.expander("Pipe Geometry", expanded=True):
        L = st.number_input("Pipe Length (L, m)", value=2300.0, min_value=1.0, step=10.0)
        D = st.number_input("Pipe Diameter (D, m)", value=0.05, min_value=0.001, step=0.005, format="%.3f")
        D_c = st.number_input("Coil Diameter (Dc, m)", value=5.0, min_value=D*2, step=0.1, help="Outer diameter of the coil loop.")
        roughness = st.number_input("Pipe Roughness (ε, m)", value=0.000045, min_value=0.0, step=0.00001, format="%.6f")
        angle_deg = st.slider("Pipe Angle (degrees from horizontal)", -90.0, 90.0, 0.0, 1.0)

    with st.expander("Fluid Properties", expanded=True):
        rho = st.number_input("Fluid Density (ρ, kg/m³)", value=1000.0, min_value=100.0, step=10.0)
        K_pl = st.number_input("Consistency Index (K, Pa.sⁿ)", value=0.065, min_value=0.001, step=0.005, format="%.4f")
        n_pl = st.number_input("Flow Behaviour Index (n)", value=0.9, min_value=0.1, max_value=2.0, step=0.05, format="%.2f", help="n<1: Shear-thinning, n=1: Newtonian, n>1: Shear-thickening")
        K_bulk = st.number_input("Fluid Bulk Modulus (K_bulk, Pa)", value=1.0e9, min_value=1e6, step=0.1e9, format="%.2e")

    with st.expander("Pipe Properties"):
        E_pipe = st.number_input("Pipe Young's Modulus (E, Pa)", value=200e9, min_value=1e9, step=10e9, format="%.2e")
        e_pipe = st.number_input("Pipe Wall Thickness (e, m)", value=0.005, min_value=0.001, step=0.001, format="%.4f")
        c1_constraint = st.number_input("Pipe Constraint Factor (c1)", value=1.0, min_value=0.1, max_value=1.5, step=0.1, help="Accounts for pipe anchoring effects on wave speed.")

    with st.expander("Simulation Control", expanded=True):
        N = st.slider("Number of Spatial Segments (N)", 10, 500, 50, 5, help="More segments increase accuracy but slow down simulation.")
        target_Cr = st.slider("Target Courant Number (Cr)", 0.1, 1.0, 0.95, 0.05, help="Stability requires Cr <= 1. dt is calculated based on this.")
        t_final = st.number_input("Final Simulation Time (s)", value=100.0, min_value=1.0, step=10.0)

    with st.expander("Boundary Conditions"):
        P_out_val = st.number_input("Outlet Pressure (P_out, Pa)", value=1.0e5, step=0.1e5, format="%.1e")

        st.subheader("Inlet Flow Rate Profile (Q_in)")
        q_params = {}
        q_params['t_ramp_start'] = st.number_input("Ramp Start Time (s)", value = 5.0, min_value=0.0)
        q_params['t_ramp_mid'] = st.number_input("Ramp Mid Time (s)", value = 35.0, min_value=q_params['t_ramp_start'])
        q_params['t_ramp_end'] = st.number_input("Ramp End Time (s)", value = 60.0, min_value=q_params['t_ramp_mid'])
        q_params['t_jump'] = st.number_input("Jump Time (s)", value = 80.0, min_value=q_params['t_ramp_end'])
        q_params['flow_min_lpm'] = st.number_input("Min Flow (L/min)", value = 0.0, min_value=0.0)
        q_params['flow_mid_lpm'] = st.number_input("Mid Flow (L/min)", value = 400.0, min_value=0.0)
        q_params['flow_max_lpm'] = st.number_input("Max Flow (L/min)", value = 600.0, min_value=0.0)
        q_params['flow_jump_lpm'] = st.number_input("Jump Flow (L/min)", value = 800.0, min_value=0.0)

    dt_to_use = 0.05
    a_est = 1000.0
    dx_est = L / N if N > 0 else 0
    try:
        if rho > 0 and K_bulk > 0 and E_pipe > 0 and e_pipe > 0 and D > 0:
            pipe_term_est = c1_constraint * K_bulk * D / (E_pipe * e_pipe)
            a_squared_est = (K_bulk / rho) / (1 + pipe_term_est)
            if a_squared_est > 0: a_est = math.sqrt(a_squared_est)
            else: st.sidebar.warning("Est. a² <= 0. Using fallback wave speed.")
        else: st.sidebar.warning("Invalid inputs for wave speed est. Using fallback.")

        if a_est > 0 and dx_est > 0:
             dt_to_use = target_Cr * dx_est / a_est
             if dt_to_use <=0:
                 st.sidebar.warning("Calculated dt <= 0. Check inputs.")
                 dt_to_use = 0.05
        else: st.sidebar.warning("Cannot calculate dt from Cr. Using fallback.")

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Estimated:** Wave Speed ≈ `{a_est:.1f}` m/s")
        st.sidebar.markdown(f"**Using:** `dt ≈ {dt_to_use:.5f}` s (for Cr={target_Cr:.2f})")
        st.sidebar.markdown(f"`dx = {dx_est:.2f}` m")
        st.sidebar.markdown("---")
    except Exception as e: st.sidebar.error(f"Error calculating dt: {e}")


# --- Main Area for Control and Results ---
sim_params = {
    'L': L, 'D': D, 'D_c': D_c, 'roughness': roughness, 'angle_deg': angle_deg,
    'rho': rho, 'K_pl': K_pl, 'n_pl': n_pl,
    'K_bulk': K_bulk, 'E_pipe': E_pipe, 'e_pipe': e_pipe, 'c1_constraint': c1_constraint,
    'N': N, 'dt': dt_to_use, 't_final': t_final,
    'P_out_val': P_out_val, 'q_in_config': q_params
}

if st.button("🚀 Run Simulation"):
    results_placeholder = st.empty()
    status_placeholder = st.empty()

    with results_placeholder.container():
        with st.spinner("⏳ Running MOC simulation..."):
            run_output = run_moc_simulation(sim_params)
            time_hist, P_hist, V_hist, x_nodes, status = run_output

    with status_placeholder.container():
        st.subheader("📊 Simulation Status & Log")
        with st.expander("Show Log Details", expanded=False):
            for msg in status:
                if msg.startswith("info:") or msg.startswith("progress:"): st.info(msg[msg.find(":")+1:].strip())
                elif msg.startswith("warning:"): st.warning(msg[msg.find(":")+1:].strip())
                elif msg.startswith("error:"): st.error(msg[msg.find(":")+1:].strip())
                else: st.write(msg)

    with results_placeholder.container():
        if not time_hist or P_hist.size == 0 or V_hist.size == 0:
            st.error("Simulation failed or produced no results. Please check the log above and input parameters.")
        else:
            st.subheader("📈 Results Visualization")
            A_pipe = math.pi * (D/2)**2
            Q_hist_array_m3s = V_hist * A_pipe
            Q_hist_array_lpm = Q_hist_array_m3s * 60 / 0.001

            tab1, tab2, tab3 = st.tabs(["Time Histories", "Spatial Profiles", "Key Metrics"])

            with tab1:
                st.markdown("#### Pressure & Flow Over Time")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=time_hist, y=P_hist[:, 0] / 1e5, name='Inlet P (bar)', mode='lines'))
                fig1.add_trace(go.Scatter(x=time_hist, y=P_hist[:, N] / 1e5, name='Outlet P (bar)', mode='lines', line=dict(dash='dash')))
                fig1.add_trace(go.Scatter(x=time_hist, y=P_hist[:, round(N/2)] / 1e5, name='Midpoint P (bar)', mode='lines', line=dict(dash='dot')))
                fig1.update_layout(xaxis_title="Time (s)", yaxis_title="Pressure (bar)")
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=time_hist, y=Q_hist_array_lpm[:, 0], name='Inlet Q (L/min)', mode='lines'))
                fig2.add_trace(go.Scatter(x=time_hist, y=Q_hist_array_lpm[:, N], name='Outlet Q (L/min)', mode='lines', line=dict(dash='dash')))
                fig2.add_trace(go.Scatter(x=time_hist, y=Q_hist_array_lpm[:, round(N/2)], name='Midpoint Q (L/min)', mode='lines', line=dict(dash='dot')))
                fig2.update_layout(xaxis_title="Time (s)", yaxis_title="Flow Rate (L/min)")
                st.plotly_chart(fig2, use_container_width=True)

            with tab2:
                st.markdown("#### Profiles Along Pipe Length")
                max_time_index = len(time_hist) - 1
                selected_time_index = st.slider("Select Time Index for Profiles", 0, max_time_index, max_time_index, help="Slide to view profiles at different simulation times.")
                selected_time = time_hist[selected_time_index]
                st.markdown(f"Displaying profiles at **t = {selected_time:.2f} s**")

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=x_nodes, y=P_hist[selected_time_index, :] / 1e5, name=f'Pressure Profile', mode='lines'))
                fig3.update_layout(title=f"Pressure Profile at t={selected_time:.2f}s", xaxis_title="Pipe Length (m)", yaxis_title="Pressure (bar)")
                st.plotly_chart(fig3, use_container_width=True)

                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=x_nodes, y=Q_hist_array_lpm[selected_time_index, :], name=f'Flow Profile', mode='lines'))
                fig4.update_layout(title=f"Flow Rate Profile at t={selected_time:.2f}s", xaxis_title="Pipe Length (m)", yaxis_title="Flow Rate (L/min)")
                st.plotly_chart(fig4, use_container_width=True)

            with tab3:
                 st.markdown("#### Summary Metrics")
                 col1, col2, col3 = st.columns(3)
                 final_P_in_bar = P_hist[-1, 0] / 1e5
                 max_P_in_bar = np.max(P_hist[:, 0]) / 1e5
                 final_dP_bar = (P_hist[-1, 0] - P_hist[-1, N]) / 1e5

                 with col1:
                     st.metric("Final Inlet Pressure (bar)", f"{final_P_in_bar:.2f}")
                     st.metric("Final Outlet Flow (L/min)", f"{Q_hist_array_lpm[-1, N]:.1f}")
                 with col2:
                     st.metric("Max Inlet Pressure (bar)", f"{max_P_in_bar:.2f}")
                     st.metric("Final Pressure Drop (bar)", f"{final_dP_bar:.2f}")
                 with col3:
                      st.metric("Final Midpoint Velocity (m/s)", f"{V_hist[-1, round(N/2)]:.3f}")
                      # Safely display Courant Number (Corrected in previous step)
                      cr_value = sim_params.get('Cr', 'N/A')
                      if isinstance(cr_value, (int, float)):
                          cr_display_str = f"{cr_value:.3f}"
                      else:
                          cr_display_str = str(cr_value)
                      st.metric("Final Courant Number", cr_display_str)


else:
    st.info("Adjust parameters in the sidebar and click '🚀 Run Simulation' to start.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for MOC Simulation.")