from numpy import *

def system_dynamics(t, x, u):
    g = 9.81
    m = 1.0
    Ixx = 0.005
    Iyy = 0.005
    Izz = 0.009

    x_pos = x[0]
    y = x[1]
    z = x[2]
    dx = x[3]
    dy = x[4]
    dz = x[5]
    phi = x[6]
    theta = x[7]
    psi = x[8]
    p = x[9]
    q = x[10]
    r = x[11]

    U1 = u[0]
    U2 = u[1]
    U3 = u[2]
    U4 = u[3]

    s_phi = sin(phi)
    c_phi = cos(phi)
    s_the = sin(theta)
    c_the = cos(theta)
    s_psi = sin(psi)
    c_psi = cos(psi)

    ddx = (1 / m) * (c_psi * s_the * c_phi + s_psi * s_phi) * U1
    ddy = (1 / m) * (s_psi * s_the * c_phi - c_psi * s_phi) * U1
    ddz = (1 / m) * (c_the * c_phi) * U1 - g
    dp = (U2 + (Iyy - Izz) * q * r) / Ixx
    dq = (U3 + (Izz - Ixx) * p * r) / Iyy
    dr = (U4 + (Ixx - Iyy) * p * q) / Izz
    dphi = p + q * s_phi * tan(theta) + r * c_phi * tan(theta)
    dtheta = q * c_phi - r * s_phi
    dpsi = q * s_phi / c_the + r * c_phi / c_the

    dxdt = array([dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]).flatten()

    return dxdt