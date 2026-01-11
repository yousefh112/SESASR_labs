import numpy as np
import math
from plot_utils import plot_obstacles, plot_trajectories, plot_velocities
from dwa import DWA
import matplotlib.pyplot as plt

def main():
    # Initialize robot's initial pose [x, y, theta] in the world frame
    init_pose = np.array([1.0, 2.0, 0.0]) # initial x, y, theta of the robot
    
    # Define the goal position [x, y] that the robot should reach
    goal_pose = np.array([8.5, 6.0]) # Goal [x, y] coordinate
    
    # Define obstacle positions [x, y] in the world frame
    obstacles = np.array([[6, 8], [2, 5], 
                          [8.5, 5], [3, 3], 
                          [5, 6], [4.5, 1.5],
                          [5, 4], [6.5, 3.5],
                          [6, 7], [3.5, 2.5],
                          [7, 3], [3, 7.5],
                          [7.5, 5], [7, 6.5],
                          [3.5, 7], [8.4, 8]]) 

    # Create a figure and axis for visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot obstacles on the map
    obst_legend = plot_obstacles(obstacles, ax)
    
    # Plot goal position as a red circle
    goal_legend = ax.scatter(goal_pose[0], goal_pose[1], marker="o", c="r", s=80, label="goal")
    
    # Plot initial robot position as a green square
    init_pose_legend = ax.scatter(init_pose[0], init_pose[1], marker="s", c="g", s=60, label="init pose")

    # Initialize DWA (Dynamic Window Approach) motion controller with parameters
    controller = DWA(
        dt=0.1,                    # Prediction time step (seconds)
        sim_time=2.0,              # Trajectory generation time horizon (seconds)
        time_granularity=0.1,      # Trajectory generation step size (seconds)
        v_samples=10,              # Number of linear velocity samples to evaluate
        w_samples=20,              # Number of angular velocity samples to evaluate
        goal_dist_tol=0.2,         # Distance tolerance to consider goal reached (meters)
        weight_angle=0.06,         # Weight for heading angle cost function
        weight_vel=0.2,            # Weight for forward velocity cost function
        weight_obs=0.04,           # Weight for obstacle avoidance cost function
        obstacles_map=obstacles,   # Known obstacle positions
        init_pose=init_pose,       # Robot's initial pose
        max_linear_acc=0.5,        # Maximum linear acceleration (m/s²)
        max_ang_acc=math.pi,       # Maximum angular acceleration (rad/s²)
        max_lin_vel=0.5,           # Maximum linear velocity (m/s)
        min_lin_vel=0.0,           # Minimum linear velocity (m/s)
        max_ang_vel=2.82,          # Maximum angular velocity (rad/s)
        min_ang_vel=-2.82,         # Minimum angular velocity (rad/s)
        radius=0.2,                # Robot body radius for collision checking (meters)
    )

    # Execute navigation to reach the goal pose
    done, robot_poses = controller.go_to_pose(goal_pose)
    
    # Print final robot position
    print("Final Robot pose: ", controller.robot.pose)
    
    # Plot the robot's trajectory path as a dashed line
    (path_legend,) = ax.plot(robot_poses[:, 0], robot_poses[:, 1], "--", label="Robot path")
    
    # Set plot title and legend
    ax.set_title("Nav to goal")
    ax.legend(handles=[goal_legend, init_pose_legend, obst_legend, path_legend])
    plt.show()

    # Plot velocity profile throughout the trajectory
    plot_velocities(robot_poses[:, 3:])

if __name__ == "__main__":
    main()
