from numpy import *


def system_dynamics(t, x, u):
    # =========================================================================
    # UNIVERSAL PLANT MODEL
    # Target: Python solve_ivp / State-Space
    # =========================================================================

    ## 1. PARAMETERS (Hard-coded constants)
    # [PARAM_START]
    mp = 1
    mw = 0.5
    lp = 0.5
    Ip = 0.1
    Iw = 0.01
    bp = 0.01
    bw = 0.05
    g = 9.81
    # [PARAM_END]

    ## 2. UNPACK STATES & INPUTS
    # [UNPACK_START]
    theta = x[0]
    dtheta = x[1]
    phi = x[2]
    dphi = x[3]
    # Assuming u is a single input, adjust if multiple inputs are expected
    input_u = u[0] if len(u) > 0 else 0  # Handle potential empty u
    # [UNPACK_END]

    ## 3. DYNAMICS / EQUATIONS OF MOTION
    # [EOM_START]
    M = Ip + Iw + mp * lp ** 2
    tau_p = -mp * g * lp * sin(theta) - bp * dtheta + input_u
    tau_w = -input_u - bw * dphi
    ddtheta = (tau_p - tau_w) / M
    ddphi = tau_w / Iw - ddtheta
    # [EOM_END]

    ## 4. DERIVATIVE VECTOR CONSTRUCTION
    # [DERIVATIVE_START]
    dxdt = array([dtheta, ddtheta, dphi, ddphi]).flatten()
    # [DERIVATIVE_END]

    return dxdt