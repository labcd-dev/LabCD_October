from numpy import *

def system_dynamics(t, x, u):
    # =========================================================================
    # UNIVERSAL PLANT MODEL
    # Target: Python solve_ivp / State-Space
    # =========================================================================

    ## 1. PARAMETERS (Hard-coded constants)
    # [PARAM_START]
    m = 50
    Iz = 5
    W = 500
    B = 510
    zb = -0.1
    zg = 0
    Zw = -20
    Zq = -5
    Zd = -10
    Mw = -3
    Mq = -8
    Md = -5
    # [PARAM_END]

    ## 2. UNPACK STATES & INPUTS
    # [UNPACK_START]
    z = x[0]
    w = x[1]
    theta = x[2]
    q = x[3]
    # Assuming u is a single input (adjust if multiple inputs)
    u_input = u[0] if len(u) > 0 else 0  # Handle potential empty u
    # [UNPACK_END]

    ## 3. DYNAMICS / EQUATIONS OF MOTION
    # [EOM_START]
    dzdt = w * cos(theta) - sin(theta)
    dwdt = (Zw * w + Zq * q + Zd * u_input + (W - B) * sin(theta)) / m
    dthetadt = q
    dqdt = (Mw * w + Mq * q + Md * u_input + (zg * W - zb * B) * sin(theta)) / Iz
    # [EOM_END]

    ## 4. DERIVATIVE VECTOR CONSTRUCTION
    # [DERIVATIVE_START]
    dxdt = array([dzdt, dwdt, dthetadt, dqdt]).flatten()
    # [DERIVATIVE_END]

    return dxdt