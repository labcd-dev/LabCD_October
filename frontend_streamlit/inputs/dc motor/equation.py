from numpy import *

def system_dynamics(t, x, u):
    R = 1.0
    L = 0.5
    Kt = 0.01
    Ke = 0.01
    J = 0.01
    b = 0.1
    TL = 0.0

    i = x[0]
    omega = x[1]
    theta = x[2]
    V = u[0]

    di = (V - R * i - Ke * omega) / L
    domega = (Kt * i - b * omega - TL) / J
    theta_dot = omega

    dxdt = array([di, domega, theta_dot])

    return dxdt