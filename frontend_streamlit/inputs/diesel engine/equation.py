from numpy import *


def system_dynamics(t, x, u):
    # =========================================================================
    # UNIVERSAL PLANT MODEL
    # Target: Python solve_ivp / State-Space
    # =========================================================================

    ## 1. PARAMETERS (Hard-coded constants)
    # [PARAM_START]
    Vm = 0.02
    Rt = 287
    Tm = 400
    Cp = 1004
    Cv = 717
    Wc = 10
    Tc = 350
    Jt = 0.05
    Kt = 0.2
    nt = 0.85
    # [PARAM_END]

    ## 2. UNPACK STATES & INPUTS
    # [UNPACK_START]
    pim = x[0]
    omega_t = x[1]
    megr = x[2]
    Wt = u[0]
    # [UNPACK_END]

    ## 3. DYNAMICS / EQUATIONS OF MOTION
    # [EOM_START]
    d_pim = Rt * Tm / Vm * (Wc - Wt - megr)
    d_omega_t = (1 / Jt) * (Cp * Tc * Wc * nt - Cp * Tm * Wt / nt - Kt * omega_t)
    d_megr = -0.5 * megr + 0.1 * pim
    # [EOM_END]

    ## 4. DERIVATIVE VECTOR CONSTRUCTION
    # [DERIVATIVE_START]
    dxdt = array([d_pim, d_omega_t, d_megr])
    # [DERIVATIVE_END]

    return dxdt