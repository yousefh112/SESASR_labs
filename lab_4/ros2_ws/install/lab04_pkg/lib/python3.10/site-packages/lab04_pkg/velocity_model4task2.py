import numpy as np
from math import cos, sin

def normalize_angle(angle):
    # Wraps the angle to the range (-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def velocity_motion_model_5state(x, u, dt):
    """
    5-state unicycle motion model (Prediction step g(mu, u, dt)):
    x = [x, y, theta, v, w] (current state)
    u = [v_cmd, w_cmd] (control input) -- UNUSED in Option A prediction
    
    Option A: Treat v, w as estimated states.
    - Use current state velocities (v, w) for kinematics.
    - Do NOT overwrite v, w with commands in prediction.
    - v, w are corrected by measurement updates (odom/IMU).
    """
    x_pos, y_pos, theta, v, w = x

    if abs(w) < 1e-4:
        # Straight-line motion
        x_new = x_pos + v * cos(theta) * dt
        y_new = y_pos + v * sin(theta) * dt
        theta_new = theta
    else:
        # Turning motion (exact unicycle)
        r = v / w
        theta_next = theta + w * dt
        x_new = x_pos - r * sin(theta) + r * sin(theta_next)
        y_new = y_pos + r * cos(theta) - r * cos(theta_next)
        theta_new = normalize_angle(theta_next)

    # Keep velocities as states (no overwrite with commands)
    v_new = v
    w_new = w

    return np.array([x_new, y_new, theta_new, v_new, w_new])

def motion_model_wrapper(mu, u, sigma_u, dt):
    """
    Wrapper to match EKF's calling convention.
    sigma_u is used by the EKF for covariance, not here.
    Controls u are accepted but ignored in Option A prediction.
    """
    return velocity_motion_model_5state(mu, u, dt)