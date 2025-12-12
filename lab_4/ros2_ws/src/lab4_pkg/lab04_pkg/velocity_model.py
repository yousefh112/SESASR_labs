import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import cos, sin, degrees, atan2, sqrt, pi
import matplotlib as mpl

arrow = u'$\u2191$'

def sample_normal_distribution(sigma_sqrd):
    return 0.5 * np.sum(np.random.default_rng().uniform(-np.sqrt(sigma_sqrd), np.sqrt(sigma_sqrd), 12))



def sample_velocity_motion_model(x, u, a, dt):
    """ Sample velocity motion model. """
    v_hat = u[0] + np.random.normal(0, a[0]*u[0]**2 + a[1]*u[1]**2)
    w_hat = u[1] + np.random.normal(0, a[2]*u[0]**2 + a[3]*u[1]**2)
    gamma_hat = np.random.normal(0, a[4]*u[0]**2 + a[5]*u[1]**2)

    r = v_hat/w_hat
    x_prime = x[0] - r*sin(x[2]) + r*sin(x[2]+w_hat*dt)
    y_prime = x[1] + r*cos(x[2]) - r*cos(x[2]+w_hat*dt)
    theta_prime = x[2] + w_hat*dt + gamma_hat*dt

    return np.array([x_prime, y_prime, theta_prime])

# --------- Added Jacobians ---------
def jacobian_Gt(x, u, dt):
    v, w = u
    theta = x[2]
    if abs(w) < 1e-9:
        return np.eye(3)
    Gt = np.array([
        [1, 0, -v/w * cos(theta) + v/w * cos(theta + w*dt)],
        [0, 1, -v/w * sin(theta) + v/w * sin(theta + w*dt)],
        [0, 0, 1]
    ])
    return Gt

def jacobian_Vt(x, u, dt):
    v, w = u
    theta = x[2]
    if abs(w) < 1e-9:
        w = 1e-9
    Vt = np.array([
        [(-sin(theta) + sin(theta + w*dt))/w,
         v*(sin(theta) - sin(theta + w*dt))/w**2 + v*dt*cos(theta + w*dt)/w],
        [(cos(theta) - cos(theta + w*dt))/w,
         -v*(cos(theta) - cos(theta + w*dt))/w**2 + v*dt*sin(theta + w*dt)/w],
        [0, dt]
    ])
    return Vt
# -----------------------------------

def main():
    plt.close('all')
    n_samples = 500
    dt = 0.5

    x = [2, 4, 0]
    u = [0.8, 0.6]
    a = [0.001, 0.01, 0.1, 0.2, 0.05, 0.05] # noise variance

    

    # --------- Added: Jacobian computation ---------
    Gx = jacobian_Gt(x, u, dt)
    Vu = jacobian_Vt(x, u, dt)
    print("Jacobian Gx:\n", Gx)
    print("Jacobian Vu:\n", Vu)
    # -----------------------------------------------

    # --------- Added: second noise set sampling -----
    a_linear = [0.2, 0.2, 0.001, 0.001, 0.001, 0.001]
    a_angular = [0.001, 0.001, 0.5, 0.5, 0.1, 0.1]




    for a_set, label, color in zip([a_linear, a_angular], ["linear noise", "angular noise"], ["r", "g"]):
        x_prime = np.zeros([n_samples, 3])
        for i in range(n_samples):
            x_prime[i,:] = sample_velocity_motion_model(x, u, a_set, dt)
        plt.scatter(x_prime[:,0], x_prime[:,1], s=15, alpha=0.4, label=label, color=color)

    plt.scatter(x[0], x[1], s=80, color='b', label='initial pose')
    plt.legend()
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Velocity motion model: linear vs angular noise")
    plt.show()
    # -----------------------------------------------

 
if __name__ == "__main__":
    main()