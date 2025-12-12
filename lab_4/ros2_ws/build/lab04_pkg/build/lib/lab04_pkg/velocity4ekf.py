import numpy as np
from math import cos, sin, sqrt
def velocity_motion_model(x, u, dt):
    v, w = u
    theta = x[2]

    if abs(w) < 1e-9:
        # straight-line motion
        x_new = x[0] + v * cos(theta) * dt
        y_new = x[1] + v * sin(theta) * dt
        theta_new = theta
    else:
        r = v / w
        x_new = x[0] - r * sin(theta) + r * sin(theta + w*dt)
        y_new = x[1] + r * cos(theta) - r * cos(theta + w*dt)
        theta_new = theta + w*dt

    return np.array([x_new, y_new, theta_new])
def velocity_motion_model_wrapper(mu, u, sigma_u, dt):
    return velocity_motion_model(mu, u, dt)



def jacobian_Gt(x, u, dt):
    """
    Evaluate Jacobian Gt w.r.t state x=[x, y, theta]
    """
    theta = x[2]
    v, w = u[0], u[1]
    r = v / w
    Gt = np.array(
        [
            [1, 0, -r * cos(theta) + r * cos(theta + w * dt)],
            [0, 1, -r * sin(theta) + r * sin(theta + w * dt)],
            [0, 0, 1],
        ]
    )

    return Gt


def jacobian_Vt(x, u, dt):
    """
    Evaluate Jacobian Vt w.r.t command u=[v,w]
    """
    theta = x[2]
    v, w = u[0], u[1]
    r = v / w
    Vt = np.array(
        [
            [
                -sin(theta) / w + sin(theta + w * dt) / w,
                dt * v * cos(theta + w * dt) / w + v * sin(theta) / w**2 - v * sin(theta + w * dt) / w**2,
            ],
            [
                -cos(theta) / w - cos(theta + w * dt) / w,
                dt * v * sin(theta + w * dt) / w - v * cos(theta) / w**2 + v * cos(theta + w * dt) / w**2,
            ],
            [0, dt],
        ]
    )

    return Vt