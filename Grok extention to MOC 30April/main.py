# main.py (v3.14 - Time-based plots show last T seconds)
# Generated on: 2025-04-30

import js
import pyodide
try:
    from pyodide.ffi import to_js
except ImportError:
    print("Warning: pyodide.ffi module not found.")
    to_js = None

try:
    import pyodide_js
except ImportError:
    print("Warning: pyodide_js module not found.")
    pyodide_js = None

import numpy as np
import math
import asyncio
import json
import traceback

def safe_js(obj):
    return js.JSON.parse(js.JSON.stringify(obj))

print("main.py: Script starting...")

# --- Simulation Constants ---
STEPS_PER_UPDATE =2
HISTORY_STORE_INTERVAL = 50
MAX_HISTORY_POINTS = 500  # Limit history to prevent slowdown

# --- Global Simulation State ---
sim_state = {
    "status": "stopped",
    "current_t": 0.0,
    "P": None, "V": None, "x_nodes": None,
    "run_params": {},
    "fluid_properties": {
        "rho": None, "K_pl": None, "n_pl": None, "K_bulk": None,
        "inlet": {"rho": 1000.0, "K_pl": 0.065, "n_pl": 0.9, "K_bulk": 1.0e9}
    },
    "log_messages": ["info: Ready."],
    "history": {"time": [], "P_in": [], "Q_in": [], "P_out": [], "Q_out": []},
    "history_step_counter": 0,
    "simulation_task": None,
    "max_pressure_bar": 0,
    "max_flow_lpm": 0
}

# --- DOM Element References ---
dom_elements = {}

# --- Core Simulation Helper Functions ---
def calculate_Re_MR(V, D, n, K, rho_fluid):
    if any(arg is None for arg in [V, D, n, K, rho_fluid]):
        add_log_message("error", f"None value in calculate_Re_MR: V={V}, D={D}, n={n}, K={K}, rho={rho_fluid}")
        return 0.0
    try:
        V = float(V); D = float(D); n = float(n); K = float(K); rho_fluid = float(rho_fluid)
    except (ValueError, TypeError) as e:
        add_log_message("error", f"Type error in calculate_Re_MR: {e}")
        return 0.0
    if abs(V) <= 1e-9: return 0.0
    if D <= 0 or K <= 0 or rho_fluid <= 0: return 0.0
    if abs(n) < 1e-9: n = 1e-9
    shear_rate_factor = (3 * n + 1) / (4 * n)
    gamma_dot_eff = shear_rate_factor * (8 * abs(V) / D)
    if abs(n - 1.0) < 1e-6: mu_app = K
    elif gamma_dot_eff <= 1e-9: mu_app = K * (1e-9)**(n - 1) if n < 1 else float('inf')
    else: mu_app = K * (gamma_dot_eff)**(n - 1)
    if mu_app <= 1e-12 or mu_app == float('inf'): return 0.0
    n_eff_kp = max(abs(n), 1e-9)
    safe_shear_factor = max(abs(shear_rate_factor), 1e-12)
    Kp_factor = (safe_shear_factor)**n_eff_kp * 8**(n_eff_kp - 1)
    K_p = K * Kp_factor
    if K_p <= 1e-12: return 0.0
    try:
        Re_MR = rho_fluid * (abs(V)**(2.0 - n)) * (D**n) / K_p
    except (OverflowError, ValueError):
        Re_MR = float('inf')
    return Re_MR

def friction_factor_turbulent_straight(Re_MR, rel_roughness, n_fluid):
    if any(arg is None for arg in [Re_MR, rel_roughness, n_fluid]):
        add_log_message("error", f"None value in friction_factor_turbulent_straight: Re={Re_MR}, rr={rel_roughness}, n={n_fluid}")
        return 0.03
    try:
        Re = float(Re_MR); rel_roughness = float(rel_roughness)
    except (ValueError, TypeError) as e:
        add_log_message("error", f"Type error in friction_factor_turbulent_straight: {e}")
        return 0.03
    RE_TURBULENT_LIMIT_NN = 4000.0
    if Re <= RE_TURBULENT_LIMIT_NN: Re = RE_TURBULENT_LIMIT_NN + 1e-3
    if rel_roughness < 0: rel_roughness = 0
    term1_eps = rel_roughness / 3.7
    term2_re = 12.0 / Re if Re > 0 else 0
    psi1 = -2.0 * np.log10(term1_eps + term2_re) if (term1_eps + term2_re) > 1e-12 else 0
    term3_re = 2.51 * psi1 / Re if Re > 0 else 0
    psi2 = -2.0 * np.log10(term1_eps + term3_re) if (term1_eps + term3_re) > 1e-12 else 0
    term4_re = 2.51 * psi2 / Re if Re > 0 else 0
    psi3 = -2.0 * np.log10(term1_eps + term4_re) if (term1_eps + term4_re) > 1e-12 else 0
    denom = (psi3 - 2.0 * psi2 + psi1)
    if abs(denom) < 1e-9 or psi1 <=0 or psi2 <=0: f_s_turb = 0.03
    else: f_s_turb = (psi1 - (psi2 - psi1)**2.0 / denom)**-2.0
    f_s_turb = max(f_s_turb, 0.008)
    return f_s_turb

def friction_factor_coiled(Re_MR, n_fluid, D_pipe, D_coil, roughness, rho_fluid, K_pl):
    default_return = (0.03, 0.0, "Error")
    args_tuple = (Re_MR, n_fluid, D_pipe, D_coil, roughness, rho_fluid, K_pl)
    if any(arg is None for arg in args_tuple):
        add_log_message("error", f"None value in friction_factor_coiled: Re={Re_MR}, n={n_fluid}, Dp={D_pipe}, Dc={D_coil}, rough={roughness}, rho={rho_fluid}, K={K_pl}")
        return default_return
    try:
        Re_MR=float(Re_MR); n_fluid=float(n_fluid); D_pipe=float(D_pipe); D_coil=float(D_coil); roughness=float(roughness)
        RE_LAMINAR_LIMIT_NN = 2100.0; RE_TURBULENT_LIMIT_NN = 4000.0
        if np.isnan(Re_MR) or np.isinf(Re_MR) or Re_MR < 0: return default_return
        f_s_lam = 64.0 / Re_MR if Re_MR > 1e-9 else float('inf'); De = 0.0
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
            regime = "Turbulent"; f_s_turb = friction_factor_turbulent_straight(Re_MR, rel_roughness, n_fluid); f_c = f_s_turb
            if De > 1e-6:
                try:
                    De_base = max(De, 1e-9); correction_turb = De_base**0.05 if De_base > 0 else 1.0
                    f_c = f_s_turb * max(1.0, correction_turb)
                except (ValueError, OverflowError): f_c = f_s_turb
            f_c = max(f_c, 0.008)
        else:
            regime = "Transition"; Re_lam_limit_calc = RE_LAMINAR_LIMIT_NN
            f_s_lam_limit = 64.0 / Re_lam_limit_calc if Re_lam_limit_calc > 1e-9 else float('inf')
            De_lam_limit = Re_lam_limit_calc * np.sqrt(max(0, D_pipe / D_coil)) if D_coil > 1e-6 else 0; f_c_lam_limit = f_s_lam_limit
            if De_lam_limit > 1e-6 and n_fluid > 0:
                try:
                    exponent_val = 1.0 - 0.5 * n_fluid; De_base = max(De_lam_limit, 1e-9)
                    De_term = De_base**exponent_val if De_base > 0 or exponent_val >= 0 else 0
                    correction_factor = 1.0 + 0.091 * (max(n_fluid, 1e-9)**0.4) * De_term
                    f_c_lam_limit = f_s_lam_limit * correction_factor; f_c_lam_limit = max(f_c_lam_limit, f_s_lam_limit)
                except (ValueError, OverflowError): pass
            f_c_lam_limit = max(0.0, f_c_lam_limit) if not np.isinf(f_c_lam_limit) else 0.0
            Re_turb_limit_calc = RE_TURBULENT_LIMIT_NN; f_s_turb_limit = friction_factor_turbulent_straight(Re_turb_limit_calc, rel_roughness, n_fluid)
            De_turb_limit = Re_turb_limit_calc * np.sqrt(max(0, D_pipe / D_coil)) if D_coil > 1e-6 else 0; f_c_turb_limit = f_s_turb_limit
            if De_turb_limit > 1e-6:
                try:
                    De_base = max(De_turb_limit, 1e-9); correction_turb = De_base**0.05 if De_base > 0 else 1.0
                    f_c_turb_limit = f_s_turb_limit * max(1.0, correction_turb)
                except (ValueError, OverflowError): pass
            f_c_turb_limit = max(f_c_turb_limit, 0.008)
            if Re_MR <= Re_lam_limit_calc + 1e-6: f_c = f_c_lam_limit
            elif Re_MR >= Re_turb_limit_calc - 1e-6: f_c = f_c_turb_limit
            elif abs(Re_turb_limit_calc - Re_lam_limit_calc) < 1e-6: f_c = (f_c_lam_limit + f_c_turb_limit) / 2.0
            elif f_c_turb_limit <= 1e-9 or f_c_lam_limit <= 1e-9: w = (Re_MR - Re_lam_limit_calc) / (Re_turb_limit_calc - Re_lam_limit_calc); f_c = f_c_lam_limit + w * (f_c_turb_limit - f_c_lam_limit)
            else:
                try:
                    log_Re_MR=np.log10(Re_MR); log_Re_lam=np.log10(Re_lam_limit_calc); log_Re_turb=np.log10(Re_turb_limit_calc)
                    log_f_lam=np.log10(f_c_lam_limit); log_f_turb=np.log10(f_c_turb_limit)
                    if abs(log_Re_turb - log_Re_lam) < 1e-9: f_c = f_c_lam_limit
                    else: log_f = log_f_lam + (log_f_turb - log_f_lam) * (log_Re_MR - log_Re_lam) / (log_Re_turb - log_Re_lam); f_c = 10**log_f
                except (ValueError, OverflowError): print(f"Warning: Math domain error during log interpolation for Re={Re_MR}. Using linear."); w = (Re_MR - Re_lam_limit_calc) / (Re_turb_limit_calc - Re_lam_limit_calc); f_c = f_c_lam_limit + w * (f_c_turb_limit - f_c_lam_limit)
            f_c = max(0.0, f_c)
        if np.isnan(f_c) or np.isinf(f_c) or np.isnan(De) or np.isinf(De): return default_return
        return f_c, De, regime
    except Exception as e: print(f"ERROR inside friction_factor_coiled: {e}"); import traceback; print(traceback.format_exc()); return default_return

def linear_interpolate(y1, y2, x_interval, x_target_rel):
    try:
        if x_interval <= 1e-9: return y1
        interp_factor = max(0.0, min(1.0, x_target_rel / x_interval))
        return y1 + (y2 - y1) * interp_factor
    except(ValueError,OverflowError): print(f"ERROR inside linear_interpolate: {interp_factor}")

def calculate_initial_conditions(Q0, P0_out, P_arr, V_arr, params, fluid_properties):
    status = []; N = params['N']; dx = params['dx']; A = params['A']; D = params['D']
    n_pl = fluid_properties['n_pl'][0]; K_pl = fluid_properties['K_pl'][0]; rho = fluid_properties['rho'][0]
    D_c = params['D_c']; roughness = params['roughness']; g = params['g']; theta_rad = params['theta_rad']
    status.append("info: Calculating initial conditions...")
    if A <= 0: raise ValueError("Pipe area must be positive.")
    V0 = Q0 / A; V_arr[:] = V0; Re0 = calculate_Re_MR(V0, D, n_pl, K_pl, rho)
    if Re0 == float('inf') or np.isnan(Re0): raise ValueError(f"Initial Reynolds number calculation failed")
    try:
        f_c0, De0, regime0 = friction_factor_coiled(Re0, n_pl, D, D_c, roughness, rho, K_pl)
        status.append(f"info: Initial State - Re_MR={Re0:.1f}, De={De0:.1f}, Regime='{regime0}', f_c={f_c0:.4f}")
    except Exception as e_ff:
        status.append(f"warning: Could not calculate initial friction factor: {e_ff}. Using default f_c=0.03")
        f_c0 = 0.03
    if D <= 0: raise ValueError("Pipe diameter must be positive.")
    friction_term0_orig = f_c0 * V0 * abs(V0) / (2.0 * D) if D > 0 else 0
    gravity_term0_orig = g * math.sin(theta_rad); RHS_steady_0_orig = - friction_term0_orig - gravity_term0_orig
    P_arr[N] = P0_out
    for j in range(N - 1, -1, -1): P_arr[j] = P_arr[j+1] - dx * RHS_steady_0_orig
    status.append(f"info: Initial conditions calculated: V0={V0:.3f} m/s, P_in={P_arr[0]/1e5:.2f} bar")
    return status

def advect_fluid_properties(V, fluid_properties, dt, dx, N):
    status_messages = []
    try:
        for prop in ['rho', 'K_pl', 'n_pl', 'K_bulk']:
            prop_new = np.zeros_like(fluid_properties[prop])
            prop_new[0] = fluid_properties['inlet'][prop]
            print(f"advect_fluid_properties: {prop}[0] set to {prop_new[0]}")
            for j in range(1, N + 1):
                V_avg = 0.5 * (V[j-1] + V[j]) if j > 0 else V[0]
                if V_avg >= 0:
                    prop_new[j] = fluid_properties[prop][j] - (V_avg * dt / dx) * (fluid_properties[prop][j] - fluid_properties[prop][j-1])
                else:
                    prop_new[j] = fluid_properties[prop][j] - (V_avg * dt / dx) * (fluid_properties[prop][j+1] - fluid_properties[prop][j]) if j < N else fluid_properties[prop][j]
                if prop == 'rho' and prop_new[j] < 100.0:
                    prop_new[j] = 100.0
                    status_messages.append(f"warning: rho[{j}] clipped to 100 kg/m³")
                elif prop == 'K_pl' and prop_new[j] < 0.001:
                    prop_new[j] = 0.001
                    status_messages.append(f"warning: K_pl[{j}] clipped to 0.001 Pa.sⁿ")
                elif prop == 'n_pl' and (prop_new[j] < 0.1 or prop_new[j] > 2.0):
                    prop_new[j] = max(0.1, min(2.0, prop_new[j]))
                    status_messages.append(f"warning: n_pl[{j}] clipped to [0.1, 2.0]")
                elif prop == 'K_bulk' and prop_new[j] < 1e6:
                    prop_new[j] = 1e6
                    status_messages.append(f"warning: K_bulk[{j}] clipped to 1e6 Pa")
            fluid_properties[prop] = prop_new
        status_messages.append("info: Fluid properties advected")
    except Exception as e:
        status_messages.append(f"error: Failed to advect fluid properties: {e}")
    return status_messages

# --- Single Step MOC Function ---
def run_moc_one_step(P, V, t, dt, current_params, fluid_properties, live_inlet_Q_m3s):
    status_messages_step = []; N = current_params['N']; dx = current_params['dx']
    g = current_params['g']; theta_rad = current_params['theta_rad']; D = current_params['D']
    A = current_params['A']; D_c = current_params['D_c']; roughness = current_params['roughness']
    E_pipe = current_params['E_pipe']; e_pipe = current_params['e_pipe']; c1 = current_params['c1_constraint']
    P_out_val = get_input_value("Pout_input") or current_params.get('P_out_val', 1.0e5)
    P_new = np.zeros_like(P); V_new = np.zeros_like(V)
    
    try:
        a = np.zeros(N + 1)
        for j in range(N + 1):
            rho_j = fluid_properties['rho'][j]; K_bulk_j = fluid_properties['K_bulk'][j]
            if rho_j > 0 and K_bulk_j > 0 and E_pipe > 0 and e_pipe > 0 and D > 0:
                pipe_term = c1 * K_bulk_j * D / (E_pipe * e_pipe)
                a_squared = (K_bulk_j / rho_j) / (1 + pipe_term)
                a[j] = math.sqrt(a_squared) if a_squared > 0 else 1000.0
            else:
                a[j] = 1000.0
                status_messages_step.append(f"warning: Invalid parameters for a[{j}], using default 1000 m/s")
        
        V_new[0] = live_inlet_Q_m3s / A if A > 0 else 0
        a_avg = 0.5 * (a[0] + a[1])
        print(f"run_moc_one_step: a[0]={a[0]:.2f}, rho[0]={fluid_properties['rho'][0]:.2f}, K_bulk[0]={fluid_properties['K_bulk'][0]:.2e}")
        P_S = linear_interpolate(P[0], P[1], dx, a_avg*dt)
        V_S = linear_interpolate(V[0], V[1], dx, a_avg*dt)
        Re_S = calculate_Re_MR(V_S, D, fluid_properties['n_pl'][0], fluid_properties['K_pl'][0], fluid_properties['rho'][0])
        f_c_S, _, _ = friction_factor_coiled(Re_S, fluid_properties['n_pl'][0], D, D_c, roughness, fluid_properties['rho'][0], fluid_properties['K_pl'][0])
        print(f"run_moc_one_step: Re_S={Re_S:.2f}, f_c_S={f_c_S:.4f}")
        FrictionTerm_S = f_c_S * V_S * abs(V_S) / (2.0 * D) if D > 0 else 0
        GravityTerm_BC = g * math.sin(theta_rad)
        P_new[0] = P_S + fluid_properties['rho'][0] * a_avg * (V_new[0] - V_S) + fluid_properties['rho'][0] * a_avg * (GravityTerm_BC + FrictionTerm_S) * dt
        print(f"run_moc_one_step: P_new[0]={P_new[0]/1e5:.2f} bar")
        P_new[N] = P_out_val
        x_R_rel_from_N_minus_1 = dx - a[N-1]*dt
        P_R_BC = linear_interpolate(P[N-1], P[N], dx, x_R_rel_from_N_minus_1)
        V_R_BC = linear_interpolate(V[N-1], V[N], dx, x_R_rel_from_N_minus_1)
        Re_R_BC = calculate_Re_MR(V_R_BC, D, fluid_properties['n_pl'][N], fluid_properties['K_pl'][N], fluid_properties['rho'][N])
        f_c_R_BC, _, _ = friction_factor_coiled(Re_R_BC, fluid_properties['n_pl'][N], D, D_c, roughness, fluid_properties['rho'][N], fluid_properties['K_pl'][N])
        FrictionTerm_R_BC = f_c_R_BC * V_R_BC * abs(V_R_BC) / (2.0 * D) if D > 0 else 0
        rho_a_safe_BC = fluid_properties['rho'][N]*a[N] + 1e-12
        V_new[N] = V_R_BC - (1.0/rho_a_safe_BC)*(P_new[N] - P_R_BC) - (GravityTerm_BC + FrictionTerm_R_BC)*dt
        GravityTerm_Loop = g * math.sin(theta_rad)
        
        for j in range(1, N):
            try:
                a_avg = 0.5 * (a[j-1] + a[j])
                x_R_rel_from_j_minus_1 = dx - a_avg*dt
                P_R = linear_interpolate(P[j-1], P[j], dx, x_R_rel_from_j_minus_1)
                V_R = linear_interpolate(V[j-1], V[j], dx, x_R_rel_from_j_minus_1)
                Re_R = calculate_Re_MR(V_R, D, fluid_properties['n_pl'][j], fluid_properties['K_pl'][j], fluid_properties['rho'][j])
                f_c_R, _, _ = friction_factor_coiled(Re_R, fluid_properties['n_pl'][j], D, D_c, roughness, fluid_properties['rho'][j], fluid_properties['K_pl'][j])
                FrictionTerm_R = f_c_R * V_R * abs(V_R) / (2.0 * D) if D > 0 else 0
                x_S_rel_from_j = a_avg*dt
                P_S = linear_interpolate(P[j], P[j+1], dx, x_S_rel_from_j)
                V_S = linear_interpolate(V[j], V[j+1], dx, x_S_rel_from_j)
                Re_S = calculate_Re_MR(V_S, D, fluid_properties['n_pl'][j], fluid_properties['K_pl'][j], fluid_properties['rho'][j])
                f_c_S, _, _ = friction_factor_coiled(Re_S, fluid_properties['n_pl'][j], D, D_c, roughness, fluid_properties['rho'][j], fluid_properties['K_pl'][j])
                FrictionTerm_S = f_c_S * V_S * abs(V_S) / (2.0 * D) if D > 0 else 0
                B_R = V_R - (GravityTerm_Loop + FrictionTerm_R)*dt
                B_S = V_S - (GravityTerm_Loop + FrictionTerm_S)*dt
                rho_a_safe_Loop = fluid_properties['rho'][j]*a_avg + 1e-12
                V_new[j] = 0.5 * ( B_R + B_S + (P_R - P_S)/rho_a_safe_Loop )
                P_new[j] = 0.5 * ( P_R + P_S + rho_a_safe_Loop*(B_R - B_S) )
                if np.isnan(P_new[j]) or np.isnan(V_new[j]): raise ValueError(f"NaN detected at node {j}, t={t+dt:.3f}.")
            except Exception as e:
                status_messages_step.append(f"error: Error at node {j}, t={t:.3f}s: {e}")
                return P, V, status_messages_step
        return P_new, V_new, status_messages_step
    except Exception as e:
        status_messages_step.append(f"error: Error during MOC step at t={t:.3f}s: {e}")
        return P, V, status_messages_step

# --- UI Update Functions ---
def add_log_message(msg_type, text):
    log_entry = f"{msg_type}: {text}"
    sim_state["log_messages"].append(log_entry)
    log_area = dom_elements.get("log_area")
    if log_area:
        max_log_lines = 20
        display_logs = sim_state["log_messages"][-max_log_lines:]
        log_area.innerText = "\n".join(reversed(display_logs))

def update_status_display():
    status_area = dom_elements.get("status_area")
    status = sim_state['status']
    current_t = sim_state['current_t']
    if status_area:
        if status == "running": status_text = f"🟢 Running... t = {current_t:.2f} s"
        elif status == "paused": status_text = f"🟡 Paused at t = {current_t:.2f} s"
        elif status == "stopped": status_text = f"🔴 Stopped at t = {current_t:.2f} s"
        elif status == "error": status_text = f"❌ Error at t = {current_t:.2f} s"
        else: status_text = "⚪ Unknown"
        status_area.innerText = status_text

def update_metrics_display():
    metrics_area = dom_elements.get("metrics_area")
    if metrics_area and sim_state["P"] is not None and sim_state["V"] is not None:
        P = sim_state["P"]; V = sim_state["V"]; N = sim_state["run_params"].get('N', 0); A = sim_state["run_params"].get('A', 0)
        if N > 0 and A > 0:
            Q_in_lpm = V[0] * A * 60000.0; Q_out_lpm = V[N] * A * 60000.0; P_in_bar = P[0] / 1e5; dP_bar = (P[0] - P[N]) / 1e5
            metrics_area.innerHTML = f"<p> Inlet P: {P_in_bar:.2f} bar | Outlet Q: {Q_out_lpm:.1f} L/min | dP: {dP_bar:.2f} bar | Inlet Q: {Q_in_lpm:.1f} L/min </p>"
        else: metrics_area.innerText = "Metrics unavailable (N or A invalid)"
    elif metrics_area: metrics_area.innerText = "Metrics unavailable (simulation not run)"

def sanitize_for_json(val):
    if isinstance(val, (list, tuple)): return [sanitize_for_json(item) for item in val]
    elif isinstance(val, dict): return {k: sanitize_for_json(v) for k, v in val.items()}
    elif isinstance(val, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(val)
    elif isinstance(val, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(val) or np.isinf(val): return None
        return float(val)
    elif isinstance(val, (np.bool_)): return bool(val)
    elif isinstance(val, (np.ndarray)): return [sanitize_for_json(item) for item in val.tolist()]
    else: return val

def create_plot_data(x, y, name, mode='lines', line_opts=None):
    trace = { 'x': sanitize_for_json(x), 'y': sanitize_for_json(y), 'name': name, 'mode': mode }
    if line_opts: trace['line'] = sanitize_for_json(line_opts)
    return trace

def update_plots():
    if sim_state["x_nodes"] is None or sim_state["P"] is None or sim_state["V"] is None:
        return
    try:
        x_nodes = sim_state["x_nodes"]
        P = sim_state["P"]
        V = sim_state["V"]
        A = sim_state["run_params"].get('A', 0)
        Q_lpm = V * A * 60000.0 if A > 0 else np.zeros_like(V)
        sim_state["max_pressure_bar"] = max(sim_state["max_pressure_bar"], np.max(P)/1e5)
        sim_state["max_flow_lpm"] = max(sim_state["max_flow_lpm"], np.max(Q_lpm))
        pressure_trace = create_plot_data(x_nodes, P / 1e5, 'Pressure')
        flow_trace = create_plot_data(x_nodes, Q_lpm, 'Flow Rate')
        profile_layout_p = {
            'title': f'Pressure Profile at t={sim_state["current_t"]:.2f}s',
            'xaxis': {'title': 'Pipe Length (m)'},
            'yaxis': {
                'title': 'Pressure (bar)',
                'range': [0, sim_state["max_pressure_bar"]],
                'autorange': False
            },
            'uirevision': 'profile_p'
        }
        profile_layout_q = {
            'title': f'Flow Rate Profile at t={sim_state["current_t"]:.2f}s',
            'xaxis': {'title': 'Pipe Length (m)'},
            'yaxis': {
                'title': 'Flow Rate (L/min)',
                'range': [0, sim_state["max_flow_lpm"]],
                'autorange': False
            },
            'uirevision': 'profile_q'
        }
        js.renderPlot('plot_pressure_profile', pyodide.ffi.to_js([pressure_trace], dict_converter=js.Object.fromEntries), pyodide.ffi.to_js(profile_layout_p, dict_converter=js.Object.fromEntries))
        js.renderPlot('plot_flow_profile', pyodide.ffi.to_js([flow_trace], dict_converter=js.Object.fromEntries), pyodide.ffi.to_js(profile_layout_q, dict_converter=js.Object.fromEntries))
        
        # Get time window for history plots
        time_window = get_input_value("time_window_slider", "range") or 30.0  # Default to 30 s if slider missing
        time_window = max(15.0, min(60.0, time_window))  # Clamp to 15-60 s
        print(f"update_plots: Time window set to {time_window} s")
        
        # Filter history data for the last T seconds
        current_t = sim_state["current_t"]
        time_min = current_t - time_window
        history_indices = [i for i, t in enumerate(sim_state["history"]["time"]) if t >= time_min]
        if history_indices:
            filtered_history = {
                key: [sim_state["history"][key][i] for i in history_indices]
                for key in sim_state["history"]
            }
        else:
            filtered_history = {key: [] for key in sim_state["history"]}
        
        # History plots
        history_layout_p = {
            'title': {'text': f'Pressure History (Last {time_window:.0f} s)'},
            'xaxis': {
                'title': {'text': 'Time (s)'},
                'range': [time_min, current_t],
                'autorange': False
            },
            'yaxis': {
                'title': {'text': 'Pressure (bar)'},
                'autorange': True
            },
            'uirevision': 'history_p'
        }
        history_layout_q = {
            'title': {'text': f'Flow Rate History (Last {time_window:.0f} s)'},
            'xaxis': {
                'title': {'text': 'Time (s)'},
                'range': [time_min, current_t],
                'autorange': False
            },
            'yaxis': {
                'title': {'text': 'Flow Rate (L/min)'},
                'autorange': True
            },
            'uirevision': 'history_q'
        }
        if len(filtered_history["time"]) > 1:
            p_in_hist_trace = create_plot_data(filtered_history["time"], np.array(filtered_history["P_in"], dtype=np.float64) / 1e5, 'Inlet P (bar)')
            p_out_hist_trace = create_plot_data(filtered_history["time"], np.array(filtered_history["P_out"], dtype=np.float64) / 1e5, 'Outlet P (bar)', line_opts={'dash': 'dash'})
            q_in_hist_trace = create_plot_data(filtered_history["time"], np.array(filtered_history["Q_in"], dtype=np.float64), 'Inlet Q (L/min)')
            q_out_hist_trace = create_plot_data(filtered_history["time"], np.array(filtered_history["Q_out"], dtype=np.float64), 'Outlet Q (L/min)', line_opts={'dash': 'dash'})
            hist_data_p = pyodide.ffi.to_js([p_in_hist_trace, p_out_hist_trace], dict_converter=js.Object.fromEntries)
            hist_data_q = pyodide.ffi.to_js([q_in_hist_trace, q_out_hist_trace], dict_converter=js.Object.fromEntries)
        else:
            hist_data_p = pyodide.ffi.to_js([], dict_converter=js.Object.fromEntries)
            hist_data_q = pyodide.ffi.to_js([], dict_converter=js.Object.fromEntries)
        js.renderPlot('plot_pressure_history', hist_data_p, pyodide.ffi.to_js(history_layout_p, dict_converter=js.Object.fromEntries))
        js.renderPlot('plot_flow_history', hist_data_q, pyodide.ffi.to_js(history_layout_q, dict_converter=js.Object.fromEntries))
    except Exception as e:
        add_log_message("error", f"Plotting error: {e}")

# --- Event Handlers ---
def get_input_value(element_id, input_type='number'):
    try:
        element = js.document.getElementById(element_id)
        if element:
            if input_type == 'number' or input_type == 'range':
                return float(element.value)
            else:
                return element.value
        else:
            add_log_message("warning", f"Element '{element_id}' not found.")
            return None
    except Exception as e:
        add_log_message("warning", f"Error getting value from '{element_id}': {e}")
        return None

def initialize_sim_state_internal(params):
    sim_state["status"] = "stopped"; sim_state["current_t"] = 0.0
    sim_state["run_params"] = params.copy()
    N = params['N']; L = params['L']; A = params.get('A', 0)
    initial_flow_lpm = params['q_in_config'].get('flow_min_lpm', 0.0); initial_Q0 = initial_flow_lpm * 0.001 / 60.0
    P0_out = params['P_out_val']
    try:
        sim_state["x_nodes"] = np.linspace(0, L, N + 1); sim_state["P"] = np.zeros(N + 1); sim_state["V"] = np.zeros(N + 1)
        sim_state["fluid_properties"]["rho"] = np.full(N + 1, sim_state["fluid_properties"]["inlet"]["rho"])
        sim_state["fluid_properties"]["K_pl"] = np.full(N + 1, sim_state["fluid_properties"]["inlet"]["K_pl"])
        sim_state["fluid_properties"]["n_pl"] = np.full(N + 1, sim_state["fluid_properties"]["inlet"]["n_pl"])
        sim_state["fluid_properties"]["K_bulk"] = np.full(N + 1, sim_state["fluid_properties"]["inlet"]["K_bulk"])
        init_params_subset = {k: params[k] for k in ['N', 'dx', 'A', 'D', 'D_c', 'roughness', 'g', 'theta_rad'] if k in params}
        init_status = calculate_initial_conditions(initial_Q0, P0_out, sim_state["P"], sim_state["V"], init_params_subset, sim_state["fluid_properties"])
        sim_state["log_messages"].extend(init_status)
        sim_state["history"] = {
            "time": [0.0],
            "P_in": [sim_state["P"][0]],
            "Q_in": [sim_state["V"][0] * A * 60000.0],
            "P_out": [sim_state["P"][N]],
            "Q_out": [sim_state["V"][N] * A * 60000.0]
        }
        sim_state["history_step_counter"] = 0
    except Exception as e:
        add_log_message("error", f"Failed to initialize arrays/conditions: {e}")
        sim_state["status"] = "error"

def reset_sim(*args, **kwargs):
    print("reset_sim called")
    if sim_state["simulation_task"] and not sim_state["simulation_task"].done():
        sim_state["simulation_task"].cancel(); add_log_message("info", "Simulation task cancelled.")
    sim_state["simulation_task"] = None; sim_state["status"] = "stopped"
    try:
        L=get_input_value("L_input"); D=get_input_value("D_input"); Dc=get_input_value("Dc_input")
        roughness=get_input_value("rough_input"); angle_deg=get_input_value("angle_input", 'range')
        rho=get_input_value("rho_widget"); K_pl=get_input_value("Kpl_input"); n_pl=get_input_value("npl_input")
        K_bulk=get_input_value("Kbulk_input"); E_pipe=get_input_value("Epipe_input"); e_pipe=get_input_value("epipe_input")
        c1=get_input_value("c1_input"); N=int(get_input_value("N_input", 'range')); target_Cr=get_input_value("Cr_input", 'range')
        t_final=get_input_value("tfinal_input"); P_out=get_input_value("Pout_input"); initial_flow=get_input_value("init_flow_input")
        if any(v is None for v in [L, D, Dc, roughness, angle_deg, rho, K_pl, n_pl, K_bulk, E_pipe, e_pipe, c1, N, target_Cr, t_final, P_out, initial_flow]):
            raise ValueError("One or more input values could not be read.")
        if N <= 0 or D <= 0: raise ValueError("N and D must be positive.")
        dx = L / N; A = math.pi * (D/2)**2; g = 9.81; theta_rad = math.radians(angle_deg)
        a_est = 1000.0; dt_to_use = 0.05
        try:
            if rho > 0 and K_bulk > 0 and E_pipe > 0 and e_pipe > 0 and D > 0:
                pipe_term_est = c1 * K_bulk * D / (E_pipe * e_pipe)
                a_squared_est = (K_bulk / rho) / (1 + pipe_term_est)
                if a_squared_est > 0: a_est = math.sqrt(a_squared_est)
            if a_est > 0 and dx > 0: dt_to_use = target_Cr * dx / a_est
            if dt_to_use <=0: dt_to_use = 0.05
        except Exception as e_dt: add_log_message("warning", f"Could not estimate dt accurately: {e_dt}")
        run_params = {
            'L': L, 'D': D, 'D_c': Dc, 'roughness': roughness, 'angle_deg': angle_deg,
            'E_pipe': E_pipe, 'e_pipe': e_pipe, 'c1_constraint': c1,
            'N': N, 'dt': dt_to_use, 't_final': t_final, 'Cr_target': target_Cr,
            'P_out_val': P_out, 'q_in_config': {'flow_min_lpm': initial_flow},
            'dx': dx, 'A': A, 'g': g, 'theta_rad': theta_rad
        }
        sim_state["fluid_properties"]["inlet"] = {'rho': rho, 'K_pl': K_pl, 'n_pl': n_pl, 'K_bulk': K_bulk}
        initialize_sim_state_internal(run_params)
        update_status_display(); update_metrics_display(); update_plots()
        add_log_message("info", "Simulation reset with new parameters.")
    except Exception as e:
        add_log_message("error", f"Failed to reset simulation: {e}")
        sim_state["status"] = "error"; update_status_display()

async def simulation_loop():
    add_log_message("info", "Simulation loop starting.")
    while True:
        if sim_state["status"] != "running":
            await asyncio.sleep(0.1)
            if sim_state["status"] != "running":
                if sim_state["simulation_task"] and sim_state["simulation_task"].cancelled():
                    add_log_message("info", "Simulation loop detected cancellation.")
                    break
                continue
        if sim_state["current_t"] >= sim_state["run_params"]["t_final"]:
            add_log_message("info", "Simulation reached t_final.")
            sim_state["status"] = "stopped"; update_status_display(); break
        steps_taken_this_batch = 0
        try:
            V = sim_state["V"]
            max_V = np.max(np.abs(V)) if V is not None else 0.0
            dx = sim_state["run_params"]["dx"]
            target_Cr = sim_state["run_params"]["Cr_target"]
            a_max = 1000.0
            for j in range(sim_state["run_params"]["N"] + 1):
                rho_j = sim_state["fluid_properties"]["rho"][j]
                K_bulk_j = sim_state["fluid_properties"]["K_bulk"][j]
                E_pipe = sim_state["run_params"]["E_pipe"]
                e_pipe = sim_state["run_params"]["e_pipe"]
                c1 = sim_state["run_params"]["c1_constraint"]
                D = sim_state["run_params"]["D"]
                if rho_j > 0 and K_bulk_j > 0 and E_pipe > 0 and e_pipe > 0 and D > 0:
                    pipe_term = c1 * K_bulk_j * D / (E_pipe * e_pipe)
                    a_squared = (K_bulk_j / rho_j) / (1 + pipe_term)
                    if a_squared > 0:
                        a_max = max(a_max, math.sqrt(a_squared))
            dt_moc = target_Cr * dx / a_max if a_max > 0 else 0.05
            dt_adv = 0.95 * dx / max_V if max_V > 1e-6 else dt_moc
            dt_run = min(dt_moc, dt_adv, 0.05)
            sim_state["run_params"]["dt"] = dt_run
            
            while steps_taken_this_batch < STEPS_PER_UPDATE and sim_state["current_t"] < sim_state["run_params"]["t_final"]:
                if sim_state["status"] != "running": break
                live_Q_lpm = get_input_value("live_flow_slider", "range")
                live_Q_m3s = (live_Q_lpm if live_Q_lpm is not None else 0.0) * 0.001 / 60.0
                # Update inlet fluid properties from UI inputs
                rho = get_input_value("rho_widget")
                K_pl = get_input_value("Kpl_input")
                n_pl = get_input_value("npl_input")
                K_bulk = get_input_value("Kbulk_input")
                if all(v is not None for v in [rho, K_pl, n_pl, K_bulk]):
                    new_inlet_props = {'rho': rho, 'K_pl': K_pl, 'n_pl': n_pl, 'K_bulk': K_bulk}
                    sim_state["fluid_properties"]["inlet"] = new_inlet_props
                    print(f"simulation_loop: Updated inlet properties: rho={rho}, K_pl={K_pl}, n_pl={n_pl}, K_bulk={K_bulk}")
                    add_log_message("info", f"Inlet properties updated: rho={rho:.1f} kg/m³, K_pl={K_pl:.3f} Pa.sⁿ, n_pl={n_pl:.2f}, K_bulk={K_bulk:.1e} Pa")
                else:
                    add_log_message("warning", "Failed to update inlet properties: invalid UI input values")
                P_curr = sim_state["P"]; V_curr = sim_state["V"]
                t_curr = sim_state["current_t"]; dt_run = sim_state["run_params"]["dt"]
                # Apply inlet properties before MOC step
                for prop in ['rho', 'K_pl', 'n_pl', 'K_bulk']:
                    sim_state["fluid_properties"][prop][0] = sim_state["fluid_properties"]["inlet"][prop]
                P_next, V_next, step_msgs = run_moc_one_step(P_curr, V_curr, t_curr, dt_run, sim_state["run_params"], sim_state["fluid_properties"], live_Q_m3s)
                sim_state["log_messages"].extend(step_msgs)
                if any("error:" in msg for msg in step_msgs):
                    sim_state["status"] = "error"
                    add_log_message("error", "Error during MOC step, stopping simulation.")
                    break
                adv_msgs = advect_fluid_properties(V_curr, sim_state["fluid_properties"], dt_run, dx, sim_state["run_params"]["N"])
                sim_state["log_messages"].extend(adv_msgs)
                if any("error:" in msg for msg in adv_msgs):
                    sim_state["status"] = "error"
                    add_log_message("error", "Error during advection, stopping simulation.")
                    break
                sim_state["P"] = P_next; sim_state["V"] = V_next; sim_state["current_t"] += dt_run
                sim_state["history_step_counter"] += 1; steps_taken_this_batch += 1
                
                if sim_state["history_step_counter"] % HISTORY_STORE_INTERVAL == 0:
                    A_hist = sim_state["run_params"].get('A', 0)
                    if A_hist > 0:
                        # Trim history to MAX_HISTORY_POINTS to prevent slowdown
                        if len(sim_state["history"]["time"]) >= MAX_HISTORY_POINTS:
                            for key in sim_state["history"]:
                                sim_state["history"][key] = sim_state["history"][key][-MAX_HISTORY_POINTS+1:]
                            add_log_message("info", f"History trimmed to {MAX_HISTORY_POINTS} points")
                        sim_state["history"]["time"].append(sim_state["current_t"])
                        sim_state["history"]["P_in"].append(P_next[0])
                        sim_state["history"]["Q_in"].append(V_next[0] * A_hist * 60000.0)
                        sim_state["history"]["P_out"].append(P_next[sim_state["run_params"]['N']])
                        sim_state["history"]["Q_out"].append(V_next[sim_state["run_params"]['N']] * A_hist * 60000.0)
            if sim_state["status"] == "error": break
            update_status_display(); update_metrics_display()
            # Update plots every 150 steps to reduce overhead
            if sim_state["history_step_counter"] % 10 == 0:
                update_plots()
            await asyncio.sleep(0.01)
        except Exception as e:
            add_log_message("error", f"Error in simulation loop: {e}")
            sim_state["status"] = "error"; update_status_display(); break
    add_log_message("info", f"Simulation loop ended with status: {sim_state['status']}")
    sim_state["simulation_task"] = None

def start_sim(*args, **kwargs):
    print("start_sim called")
    if sim_state["status"] in ["stopped", "paused"]:
        is_resume = sim_state["current_t"] > 0.0 and sim_state["status"] == "paused"
        sim_state["status"] = "running"
        add_log_message("info", "Simulation starting/resuming...")
        update_status_display()
        if not is_resume and (sim_state["P"] is None or not sim_state["run_params"]):
            add_log_message("warning", "State not initialized. Please Reset first.")
            reset_sim()
            if sim_state["status"] == "error": return
        if sim_state.get("simulation_task") is None or sim_state["simulation_task"].done():
            sim_state["simulation_task"] = asyncio.create_task(simulation_loop())
            add_log_message("info", "Simulation task created/restarted.")
        else:
            add_log_message("info", "Simulation task already running (or finishing cancellation).")
    else:
        add_log_message("warning", "Simulation already running or in error state.")

def pause_sim(*args, **kwargs):
    print("pause_sim called")
    if sim_state["status"] == "running":
        sim_state["status"] = "paused"
        add_log_message("info", "Simulation paused.")
        update_status_display()
    else:
        add_log_message("info", "Simulation not running.")

# --- Initialization Code ---
async def setup():
    print("setup: Getting DOM elements...")
    element_ids = [
        "L_input", "D_input", "Dc_input", "rough_input", "angle_input",
        "rho_widget", "Kpl_input", "npl_input", "Kbulk_input",
        "Epipe_input", "epipe_input", "c1_input",
        "N_input", "Cr_input", "tfinal_input", "Pout_input", "init_flow_input",
        "live_flow_slider", "time_window_slider",  # Added for time window
        "start_btn", "pause_btn", "reset_btn",
        "status_area", "log_area", "metrics_area",
        "plot_pressure_profile", "plot_flow_profile",
        "plot_pressure_history", "plot_flow_history"
    ]
    for id in element_ids:
        dom_elements[id] = js.document.getElementById(id)
        print(f"setup: Checking {id}: {'Found' if dom_elements[id] else 'Not Found'}")
        if not dom_elements[id]:
            add_log_message("warning", f"Element with ID '{id}' not found.")
        elif id == "time_window_slider" and dom_elements[id]:
            add_log_message("info", f"Time window slider initialized to {dom_elements[id].value} s")
    print("setup: Initial reset...")
    reset_sim()
    print("setup: Complete.")
    
asyncio.create_task(setup())
print("main.py: Script finished initial execution.")