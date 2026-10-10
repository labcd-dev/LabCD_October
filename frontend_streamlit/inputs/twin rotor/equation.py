from numpy import *


def system_dynamics(t, x, u):
    # =========================================================================
    # UNIVERSAL PLANT MODEL
    # Target: Python solve_ivp / State-Space
    # =========================================================================

    ## 1. PARAMETERS (Hard-coded constants)
    # [PARAM_START]
    J_vertical = 0.05  # Moment of inertia around pitch axis [kg*m**2]
    J_horizontal = 0.04  # Moment of inertia around yaw axis [kg*m**2]
    a1 = 0.012  # Main rotor lift coefficient
    a2 = 0.010  # Tail rotor thrust coefficient
    B_pitch = 0.01  # Viscous friction coefficient for pitch
    B_yaw = 0.015  # Viscous friction coefficient for yaw
    MgL = 0.3  # Gravitational momentum parameter [N*m]
    K_gyroscopic = 0.005  # Gyroscopic cross-coupling torque coefficient
    # [PARAM_END]

    ## 2. UNPACK STATES & INPUTS
    # [UNPACK_START]
    theta = x[0]  # Pitch angle [rad]
    dtheta = x[1]  # Pitch angular velocity [rad/s]
    psi = x[2]  # Yaw angle [rad]
    dpsi = x[3]  # Yaw angular velocity [rad/s]
    U_main = u[0]  # Control voltage/PWM to main motor
    U_tail = u[1]  # Control voltage/PWM to tail motor
    # [UNPACK_END]

    ## 3. DYNAMICS / EQUATIONS OF MOTION
    # [EOM_START]
    Tau_main = a1 * U_main  # Main thrust force
    Tau_tail = a2 * U_tail  # Tail thrust force
    Tau_cross = K_gyroscopic * Tau_main * dtheta  # Gyroscopic cross-coupling torque

    ddtheta = (Tau_main - MgL * cos(theta) - B_pitch * dtheta) / J_vertical
    ddpsi = (Tau_tail - B_yaw * dpsi + Tau_cross) / J_horizontal
    # [EOM_END]

    ## 4. DERIVATIVE VECTOR CONSTRUCTION
    # [DERIVATIVE_START]
    dxdt = array([dtheta, ddtheta, dpsi, ddpsi]).reshape(-1)
    # [DERIVATIVE_END]

    return dxdt