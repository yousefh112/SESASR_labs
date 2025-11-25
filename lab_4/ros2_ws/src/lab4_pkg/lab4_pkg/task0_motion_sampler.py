#!/usr/bin/env python3

"""
task0_motion_sampler.py

This script runs the velocity motion model sampling required for Task 0.
It imports the sampling function from 'velocity_motion_model.py' and
generates two plots for two different noise configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.markers import MarkerStyle

# Import the sampling function from your other file
from lab4_pkg.velocity_motion_model import sample_velocity_motion_model

def plot_samples(ax, samples, initial_pose, title):
    """Helper function to plot samples."""
    # Plot samples
    ax.scatter(samples[:, 0], samples[:, 1], c='r', marker='.', label='Sampled Poses', alpha=0.5)
    
    # Plot initial pose
    arrow = u'$\u2191$'
    rotated_marker = MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(initial_pose[2]) - 90)
    ax.scatter(initial_pose[0], initial_pose[1], marker=rotated_marker, s=200, facecolors='none', edgecolors='b', label='Initial Pose')
    
    ax.set_title(title)
    ax.set_xlabel("x-position [m]")
    ax.set_ylabel("y-position [m]")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

def main():
    N_SAMPLES = 500  #
    DT = 0.5  # Time step (e.g., 0.5 seconds)

    # Initial state [x, y, theta]
    x_initial = np.array([2.0, 4.0, 0.0])
    # Fixed command [v, w]
    u = np.array([0.8, 0.6])

    # Noise Set 1: High Angular Uncertainty [cite: 28]
    # (a3, a4, a5, a6 are large)
    a_angular = np.array([0.001, 0.001, 0.1, 0.1, 0.05, 0.05])

    # Noise Set 2: High Linear Uncertainty [cite: 28]
    # (a1, a2 are large)
    a_linear = np.array([0.1, 0.1, 0.001, 0.001, 0.001, 0.001])

    # --- Run Sampling ---
    x_primes_angular = np.zeros([N_SAMPLES, 3])
    x_primes_linear = np.zeros([N_SAMPLES, 3])

    for i in range(N_SAMPLES):
        x_primes_angular[i, :] = sample_velocity_motion_model(x_initial, u, a_angular, DT)
        x_primes_linear[i, :] = sample_velocity_motion_model(x_initial, u, a_linear, DT)

    print(f"Generated {N_SAMPLES} samples for two noise sets.")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    plot_samples(ax1, x_primes_angular, x_initial, "High Angular Uncertainty")
    plot_samples(ax2, x_primes_linear, x_initial, "High Linear Uncertainty")

    plt.suptitle("Velocity Motion Model Sampling (Task 0)")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()