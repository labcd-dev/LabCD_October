from numpy import *

def system_dynamics(t, x, u):
    # =========================================================================
    # UNIVERSAL PLANT MODEL
    # Target: Python solve_ivp / State-Space
    #
    # Structure:
    # x = [state1, state2, ..., stateN]
    # u = [input1, input2, ..., inputM]
    # =========================================================================
    ## 1. PARAMETERS (Hard-coded constants)
    # [LLM: Define all coefficients, mass, inertia, or gains here]
    # [PARAM_START]
    m = 1000  # mass[kg]
    Iy = 3000  # pitch moment of inertia[kg*m**2]
    S = 16  # wing area[m**2]
    c = 1.5  # mean aerodynamic chord[m]
    rho = 1.225  # air density[kg/m**3]
    V = 50  # airspeed[m/s]
    CL_alpha = 4.5
    CM_alpha = -0.8
    CM_q = -15
    CM_delta = -1.2
    g = 9.81  # gravity[m/s^2]
    # [PARAM_END]

    ## 2. UNPACK STATES & INPUTS
    # [LLM: Mapping for readability]
    # [UNPACK_START]
    theta = x[0]  # pitch angle
    q = x[1]  # pitch rate
    w = x[2]  # vertical velocity
    delta_e = u[0]  # elevator deflection
    # [UNPACK_END]

    ## 3. DYNAMICS / EQUATIONS OF MOTION
    # [LLM: Implement the differential equations]
    # [EOM_START]
    alpha = arctan(w / V)  # angle of attack
    Q = 0.5 * rho * V**2  # dynamic pressure
    L = Q * S * CL_alpha * alpha
    M = Q * S * c * (CM_alpha * alpha + CM_q * q * c / (2*V) + CM_delta * delta_e)
    theta_dot = q
    q_dot = M / Iy
    w_dot = (L * cos(theta) - m * g * sin(theta)) / m - V * q
    # [EOM_END]

    ## 4. DERIVATIVE VECTOR CONSTRUCTION
    # [LLM: Must be a vector/array matching the state order]
    # [DERIVATIVE_START]
    dxdt = array([theta_dot, q_dot, w_dot])
    # [DERIVATIVE_END]
    return dxdt