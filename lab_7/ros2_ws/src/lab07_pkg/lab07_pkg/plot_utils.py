import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import colors


def plot_obstacles(obstacles, ax):
    # Create a scatter plot of obstacle positions on the given axes
    obst_legend = ax.scatter(obstacles[:, 0], obstacles[:, 1], marker="o", c="k", s=50, label="obstacles")
    return obst_legend

def plot_trajectories(paths, ax, best_path_idx=22, N=5):
    # Get the total number of paths
    n_paths = paths.shape[0]
    
    # Plot every N-th trajectory to avoid clutter
    for i in range(0, n_paths, N):
        (sim_paths_legend,) = ax.plot(paths[i, :, 0], paths[i, :, 1], linewidth=0.8, label="sim paths")

    # Plot the best path in red dashed line with higher visibility
    (best_path_legend,) = ax.plot(paths[best_path_idx, :, 0], paths[best_path_idx, :, 1], "r--", linewidth=2, label="best path")
    
    return best_path_legend

def plot_velocities(velocities):
    # Create time array based on number of velocity samples
    t = np.arange(velocities.shape[0])
    
    # Create a figure with 2 subplots stacked vertically
    fig, axs = plt.subplots(2, 1, layout='constrained')
    
    # Plot linear velocities in the first subplot
    axs[0].plot(t, velocities[:, 0])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('v [m/s]')
    axs[0].set_title("Linear velocity")
    axs[0].grid(True)
    axs[0].legend(['v'])

    # Plot angular velocities in the second subplot
    axs[1].plot(t, velocities[:, 1])
    axs[1].set_ylabel('w [rad/s]')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title("Angular velocity")
    axs[1].grid(True)
    axs[1].legend(['w'])
    
    # Display the plot
    plt.show()