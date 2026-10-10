from numpy import *


def system_dynamics(t, x, u):
    # =========================================================================
    # UNIVERSAL PLANT MODEL
    # Target: Python solve_ivp / State-Space
    # =========================================================================

    ## 1. PARAMETERS (Hard-coded constants)
    # [PARAM_START]
    q = 1
    V = 1
    CA0 = 1
    T0 = 350
    Tc0 = 300
    k0 = 7.2e10
    E_R = 8750
    dH = -2e5
    rho = 1000
    Cp = 0.239
    UA = 5e4
    rhoc = 1000
    Cpc = 1
    Vc = 1
    # [PARAM_END]

    ## 2. UNPACK STATES & INPUTS
    # [UNPACK_START]
    CA = x[0]
    T = x[1]
    Tc = x[2]
    # [UNPACK_END]

    ## 3. DYNAMICS / EQUATIONS OF MOTION
    # [EOM_START]
    r = k0 * exp(-E_R / T) * CA
    dCA_dt = q / V * (CA0 - CA) - r
    dT_dt = q / V * (T0 - T) - dH / (rho * Cp) * r + UA / (V * rho * Cp) * (Tc - T)
    dTc_dt = u[0] / Vc * (Tc0 - Tc) + UA / (Vc * rhoc * Cpc) * (T - Tc)
    # [EOM_END]

    ## 4. DERIVATIVE VECTOR CONSTRUCTION
    # [DERIVATIVE_START]
    dxdt = array([dCA_dt, dT_dt, dTc_dt])
    # [DERIVATIVE_END]

    return dxdt