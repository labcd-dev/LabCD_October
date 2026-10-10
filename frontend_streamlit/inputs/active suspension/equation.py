from numpy import *

def system_dynamics(t, x, u):
    # =========================================================================
    # UNIVERSAL PLANT MODEL
    # Target: Python solve_ivp / State-Space
    # =========================================================================

    ## 1. PARAMETERS (Hard-coded constants)
    # [PARAM_START]
    ms = 300
    mu = 60
    ks = 16000
    kt = 190000
    bs = 1000
    # [PARAM_END]

    ## 2. UNPACK STATES & INPUTS
    # [UNPACK_START]
    zs = x[0]
    vs = x[1]
    zu = x[2]
    vu = x[3]
    # [UNPACK_END]

    ## 3. DYNAMICS / EQUATIONS OF MOTION
    # [EOM_START]
    d_zs = vs
    d_vs = (-ks*(zs-zu) - bs*(vs-vu) + u[0]) / ms
    d_zu = vu
    d_vu = (ks*(zs-zu) + bs*(vs-vu) - kt*zu - u[0]) / mu
    # [EOM_END]

    ## 4. DERIVATIVE VECTOR CONSTRUCTION
    # [DERIVATIVE_START]
    dxdt = array([d_zs, d_vs, d_zu, d_vu])
    # [DERIVATIVE_END]

    return dxdt