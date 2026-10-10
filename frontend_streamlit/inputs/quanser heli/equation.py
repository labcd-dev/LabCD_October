from numpy import *

def system_dynamics(t, x, u):
    m = 1.3872
    g = 9.81
    B_p = 0.8
    B_y = 0.318
    K_pp = 0.2040
    K_yy = 0.0720
    K_py = 0.0068
    K_yp = 0.0219
    J_p = 0.0178
    J_y = 0.0084
    l_cm = 0.186
    J_Tp = J_p + m * l_cm**2
    J_Ty = J_y + m * l_cm**2

    pitch = x[0]
    yaw = x[1]
    dpitch = x[2]
    dyaw = x[3]

    u1 = u[0]
    u2 = u[1]

    Tp = K_pp * u1 + K_py * u2
    Ty = K_yp * u1 + K_yy * u2

    d_pitch = dpitch
    d_yaw = dyaw
    dd_pitch = (Tp - B_p * dpitch - m * g * l_cm * sin(pitch)) / J_Tp
    dd_yaw = (Ty - B_y * dyaw) / J_Ty

    dxdt = array([d_pitch, d_yaw, dd_pitch, dd_yaw]).flatten()

    return dxdt