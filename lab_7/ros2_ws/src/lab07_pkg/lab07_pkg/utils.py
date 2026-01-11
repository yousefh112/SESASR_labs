import numpy as np
import math


class Differential_drive_robot():
    def __init__(self, 
                init_pose,
                max_linear_acc = 0.8,
                max_ang_acc = 100 * math.pi /180,
                max_lin_vel = 1.0, # m/s
                min_lin_vel = 0.0, # m/s
                max_ang_vel = 3.0, # rad/s 
                min_ang_vel = -3.0, # rad/s 
                radius = 0.3, # radius for circular robot
                ):
        
        # Initialize robot state
        self.pose = init_pose  # Current position and orientation [x, y, theta]
        self.vel = np.array([0.0, 0.0])  # Current velocity [linear, angular]
        
        # Kinematic constraints
        self.max_linear_acc = max_linear_acc  # Maximum linear acceleration (m/s²)
        self.max_ang_acc = max_ang_acc  # Maximum angular acceleration (rad/s²)
        self.max_lin_vel = max_lin_vel  # Maximum linear velocity (m/s)
        self.min_lin_vel = min_lin_vel  # Minimum linear velocity (m/s)
        self.max_ang_vel = max_ang_vel  # Maximum angular velocity (rad/s)
        self.min_ang_vel = min_ang_vel  # Minimum angular velocity (rad/s)

        # Robot physical properties
        self.radius = radius  # Collision radius for circular robot model

        # Trajectory history initialization - stores [x, y, theta, lin_vel, ang_vel]
        self.trajectory = np.array([init_pose[0], init_pose[1], init_pose[2], 0.0, 0.0]).reshape(1, -1)

    def update_state(self, u, dt):
        """
        Update robot state based on differential drive kinematics.
        
        Args:
            u: Control input [linear_velocity, angular_velocity]
            dt: Time step (seconds)
        
        Returns:
            Updated pose [x, y, theta]
        """

        # Convert control input to numpy array if necessary
        if u is list:
            u = np.array(u)

        self.vel = u

        # Apply differential drive kinematic equations
        next_x = self.pose[0] + self.vel[0] * math.cos(self.pose[2]) * dt
        next_y = self.pose[1] + self.vel[0] * math.sin(self.pose[2]) * dt
        next_th = self.pose[2] + self.vel[1] * dt
        self.pose = np.array([next_x, next_y, next_th])

        # Append new state to trajectory history
        traj_state = np.array([next_x, next_y, next_th, self.vel[0], self.vel[1]]).reshape(1, -1)
        self.trajectory = np.concatenate([self.trajectory, traj_state], axis=0)

        return self.pose

def calc_nearest_obs(robot_pose, obstacles, obstacle_max_dist=3):
    """
    Filter obstacles within sensor range for local obstacle avoidance.
    
    Args:
        robot_pose: Current robot position [x, y, theta]
        obstacles: List of obstacle positions
        obstacle_max_dist: Maximum detection distance (meters)
    
    Returns:
        Array of nearby obstacles within max distance
    """
    nearest_obs = []
    
    for obs in obstacles:
        # Calculate Euclidean distance between robot and obstacle
        temp_dist_to_obs = np.linalg.norm(robot_pose[0:2]-obs)

        # Keep only obstacles within detection range
        if temp_dist_to_obs < obstacle_max_dist :
            nearest_obs.append(obs)

    return np.array(nearest_obs)

def normalize_angle(theta):
    """
    Normalize angle to range [-π, π).
    
    Args:
        theta: Angle in radians (scalar or array)
    
    Returns:
        Normalized angle in range [-π, π)
    """
    # Force angle into range [0, 2π)
    theta = theta % (2 * np.pi)
    
    if np.isscalar(theta):
        # Shift angles greater than π to negative range
        if theta > np.pi:
            theta -= 2 * np.pi
    else:
        # Handle array of angles
        theta_ = theta.copy()
        theta_[theta>np.pi] -= 2 * np.pi
        return theta_
    
    return theta

def normalize(arr: np.ndarray):
    """
    Normalize array values to range [0, 1].
    
    Args:
        arr: Input array
    
    Returns:
        Normalized array, or zero array if all values are equal
    """
    # Avoid division by zero if array has no range
    if np.isclose(np.max(arr) - np.min(arr), 0.0):
        return np.zeros_like(arr)
    else:
        # Min-max normalization: (x - min) / (max - min)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def sigmoid(x: np.ndarray):
    """
    Compute sigmoid activation function: 1 / (1 + e^(-x)).
    Useful for smooth, non-linear transformations in obstacle avoidance.
    
    Args:
        x: Input array of values
    
    Returns:
        Sigmoid activation output in range (0, 1)
    """
    return 1/(1 + np.exp(-x))