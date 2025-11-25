#!/usr/bin/env python3

"""
task0_landmark_sampler.py

This script runs the landmark measurement model sampling required for Task 0.
It imports the sampling function from 'landmark_model.py' and
generates a plot of 1000 sampled poses based on a single measurement.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.markers import MarkerStyle

# Import functions from your other file
from lab4_pkg.landmark_model import landmark_model_sample_pose, landmark_range_bearing_sensor

def main():
    N_SAMPLES = 1000  # [cite: 35]

    # --- Setup ---
    # A "true" robot pose to generate a realistic measurement
    robot_pose_true = np.array([0., 0., math.pi/4])
    # A known landmark position
    landmark_pos = np.array([5., 2.])
    # Measurement noise standard deviation [sigma_r, sigma_phi]
    sigma = np.array([0.3, math.pi/24])

    print("Simulating one measurement (z)...")
    # Simulate a single, noisy measurement z = [r, phi]
    z = landmark_range_bearing_sensor(robot_pose_true, landmark_pos, sigma=np.array([0.0, 0.0])) # Use 0 noise for a "clean" z
    
    if z is None:
        print("Initial pose is outside sensor range/FOV. Exiting.")
        return

    print(f"Landmark at {landmark_pos}")
    print(f"True robot pose at {np.round(robot_pose_true, 2)}")
    print(f"Simulated measurement z = [r, phi]: {np.round(z, 2)}")
    print(f"Sampling {N_SAMPLES} poses based on this measurement...")

    # --- Run Sampling ---
    sampled_poses = np.zeros([N_SAMPLES, 3])
    for i in range(N_SAMPLES):
        # Sample a pose *given* the measurement z [cite: 35]
        sampled_poses[i, :] = landmark_model_sample_pose(z, landmark_pos, sigma)

    # --- Plotting ---
    plt.figure(figsize=(10, 10))
    
    # Plot sampled poses
    plt.scatter(sampled_poses[:, 0], sampled_poses[:, 1], c='b', marker='.', label='Sampled Poses', alpha=0.3)

    # Plot landmark
    plt.scatter(landmark_pos[0], landmark_pos[1], marker='*', s=200, c='k', label='Landmark')

    # Plot "true" pose
    arrow = u'$\u2191$'
    rotated_marker = MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(robot_pose_true[2]) - 90)
    plt.scatter(robot_pose_true[0], robot_pose_true[1], marker=rotated_marker, s=250, facecolors='none', edgecolors='g', label='"True" Pose')

    plt.title(f"Landmark Model Pose Sampling (Task 0) - {N_SAMPLES} Samples")
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()