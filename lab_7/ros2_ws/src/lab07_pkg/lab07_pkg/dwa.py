import numpy as np
from utils import Differential_drive_robot, normalize_angle, normalize, calc_nearest_obs

# DWA (Dynamic Window Approach) Controller
# This module implements a DWA controller for robot navigation simulation.
# It uses a simplified differential drive robot model (defined in utils.py).
# Core components: trajectory generation, evaluation functions (heading, velocity, obstacle avoidance),
# and the main control loop in go_to_pose().

class DWA():
    def __init__(self,
                 dt = 0.1,
                 sim_time = 2.0,                    # Total time horizon for trajectory rollout simulation
                 time_granularity = 0.1,            # Time step for trajectory simulation
                 v_samples = 10,                    # Number of linear velocity samples in dynamic window
                 w_samples = 20,                    # Number of angular velocity samples in dynamic window
                 goal_dist_tol = 0.3,               # Distance threshold for goal reached condition
                 collision_tol = 0.3,               # Safety margin for collision detection
                 weight_angle = 0.04,               # Weight for heading angle objective
                 weight_vel = 0.2,                  # Weight for velocity objective
                 weight_obs = 0.1,                  # Weight for obstacle avoidance objective
                 obstacles_map = None,              # List of obstacle poses in the map
                 **kwargs                           # Additional robot parameters
                 ):

        self.dt = dt
        self.sim_step = round(sim_time / time_granularity)
        self.robot = Differential_drive_robot(**kwargs)
        self.obstacles = obstacles_map 
        self.goal_dist_tol = goal_dist_tol
        self.collision_tol = collision_tol

        self.v_samples = v_samples
        self.w_samples = w_samples

        # Define evaluation function weights
        self.weight_angle = weight_angle
        self.weight_vel = weight_vel
        self.weight_obs = weight_obs

        self.obstacle_max_dist = 3             # Local costmap radius
        self.max_num_steps = 300               # Maximum control steps before timeout
        self.feedback_rate = 50                # Steps between status updates
        self.obst_tolerance = 0.5

    def go_to_pose(self, goal_pose):
        """
        Navigate the robot to the target goal pose using DWA control.
        
        Args:
            goal_pose: Target goal position [x, y]
            
        Returns:
            success: Boolean indicating if goal was reached
            trajectory: List of robot poses along the executed path
        """
        if goal_pose is list:
            goal_pose = np.array(goal_pose)

        success = False
        dist_to_goal = np.linalg.norm(self.robot.pose[0:2] - goal_pose)

        print("Initial distance to goal: ", dist_to_goal)
        print("Initial Robot pose: ", self.robot.pose)

        steps = 1
        while steps <= self.max_num_steps:
            # 1. Check if goal has been reached
            dist_to_goal = np.linalg.norm(self.robot.pose[0:2] - goal_pose)
            if dist_to_goal < self.goal_dist_tol:
                success = True
                print("Goal reached!")
                break

            # 2. Get new sensor observations for obstacle detection
            # Note: No lidar currently used in this simulation

            # 3. Compute velocity command using DWA algorithm
            u = self.compute_cmd(goal_pose, self.robot.pose, self.obstacles)

            # 4. Update robot state with computed command
            pose = self.robot.update_state(u, self.dt)
            
            # 5. Provide periodic feedback on task progress
            if steps % self.feedback_rate == 0:
                dist_to_goal = np.linalg.norm(self.robot.pose[0:2] - goal_pose)
                print("Current distance to goal ", dist_to_goal, " at step ", steps)
                print("Current Robot pose: ", pose)

            steps += 1

        if steps > self.max_num_steps:
            print("Timeout! Goal not reached.")

        return success, self.robot.trajectory
    
    def compute_cmd(self, goal_pose, robot_state, obstacles):
        """
        Compute the optimal velocity command using DWA algorithm.
        
        Args:
            goal_pose: Target goal position
            robot_state: Current robot state [x, y, theta]
            obstacles: List of obstacle positions
            
        Returns:
            u: Optimal velocity command [v, w] (linear and angular velocity)
        """
        # Generate all candidate trajectories within dynamic window
        paths, velocities = self.get_trajectories(robot_state)

        # Evaluate trajectories and select the best one
        opt_idx = self.evaluate_paths(paths, velocities, goal_pose, robot_state, obstacles)
        u = velocities[opt_idx]
        return u

    def get_trajectories(self, robot_pose): 
        """
        Generate all candidate trajectories within the dynamic window.
        Each trajectory is simulated with constant velocity (v, w) over sim_time.
        
        Args:
            robot_pose: Current robot state [x, y, theta]
            
        Returns:
            sim_paths: Array of shape (n_paths, sim_step, state_dim) containing trajectory points
            velocities: Array of shape (n_paths, 2) containing velocity commands [v, w]
        """
        # Calculate reachable velocity range considering kinematic constraints
        min_lin_vel, max_lin_vel, min_ang_vel, max_ang_vel = self.compute_dynamic_window(self.robot.vel)
        
        # Sample linear and angular velocities uniformly within dynamic window
        w_values = np.linspace(min_ang_vel, max_ang_vel, self.w_samples)
        v_values = np.linspace(min_lin_vel, max_lin_vel, self.v_samples)

        # Create all combinations of velocity samples
        n_paths = w_values.shape[0] * v_values.shape[0]
        sim_paths = np.zeros((n_paths, self.sim_step, robot_pose.shape[0]))
        velocities = np.zeros((n_paths, 2))

        # Generate meshgrid of all velocity combinations
        vv, ww = np.meshgrid(v_values, w_values)
        velocities = np.dstack([vv, ww]).reshape(n_paths, 2)
        
        # Simulate trajectories for each velocity command
        sim_paths = self.simulate_paths(n_paths, robot_pose, velocities)

        return sim_paths, velocities
    
    def simulate_paths(self, n_paths, pose, u):
        """
        Simulate robot trajectories under constant velocity commands.
        Uses the differential drive kinematic model.
        
        Args:
            n_paths: Number of trajectories to simulate
            pose: Initial robot state [x, y, theta]
            u: Velocity commands array of shape (n_paths, 2) with [v, w] for each path
            
        Returns:
            sim_paths: Array of shape (n_paths, sim_step, 3) containing simulated states
        """
        sim_paths = np.zeros((n_paths, self.sim_step, pose.shape[0]))
        sim_paths[:, 0] = pose.copy()

        # Simulate each trajectory step using kinematic model
        for i in range(1, self.sim_step):
            # x(t+1) = x(t) + v*cos(theta)*dt
            sim_paths[:, i, 0] = sim_paths[:, i - 1, 0] + u[:, 0] * np.cos(sim_paths[:, i - 1, 2]) * self.dt
            # y(t+1) = y(t) + v*sin(theta)*dt
            sim_paths[:, i, 1] = sim_paths[:, i - 1, 1] + u[:, 0] * np.sin(sim_paths[:, i - 1, 2]) * self.dt
            # theta(t+1) = theta(t) + w*dt
            sim_paths[:, i, 2] = sim_paths[:, i - 1, 2] + u[:, 1] * self.dt

        return sim_paths

    def compute_dynamic_window(self, robot_vel): 
        """
        Calculate the dynamic window of feasible velocities.
        The window is constrained by robot kinematic limits and current velocity.
        
        Args:
            robot_vel: Current robot velocity [v, w]
            
        Returns:
            min_vel: Minimum feasible linear velocity
            max_vel: Maximum feasible linear velocity
            min_ang_vel: Minimum feasible angular velocity
            max_ang_vel: Maximum feasible angular velocity
        """
        # Calculate linear velocity bounds based on acceleration limits
        min_vel = robot_vel[0] - self.dt * self.robot.max_linear_acc
        max_vel = robot_vel[0] + self.dt * self.robot.max_linear_acc
        
        # Enforce robot's absolute velocity limits
        if min_vel < self.robot.min_lin_vel:
            min_vel = self.robot.min_lin_vel
        if max_vel > self.robot.max_lin_vel:
            max_vel = self.robot.max_lin_vel

        # Calculate angular velocity bounds based on acceleration limits
        min_ang_vel = robot_vel[1] - self.dt * self.robot.max_ang_acc
        max_ang_vel = robot_vel[1] + self.dt * self.robot.max_ang_acc
        
        # Enforce robot's absolute angular velocity limits
        if min_ang_vel < self.robot.min_ang_vel:
            min_ang_vel = self.robot.min_ang_vel
        if max_ang_vel > self.robot.max_ang_vel:
            max_ang_vel = self.robot.max_ang_vel

        return min_vel, max_vel, min_ang_vel, max_ang_vel


    def evaluate_paths(self, paths, velocities, goal_pose, robot_pose, obstacles):
        """
        Evaluate candidate trajectories using a weighted objective function.
        J = w_h * heading_score + w_v * velocity_score + w_o * obstacle_score
        
        Args:
            paths: Candidate trajectories
            velocities: Velocity commands for each trajectory
            goal_pose: Target goal position
            robot_pose: Current robot state
            obstacles: List of obstacle positions
            
        Returns:
            opt_idx: Index of the best trajectory
        """
        # Find nearest obstacle for efficiency
        nearest_obs = calc_nearest_obs(robot_pose, obstacles)

        # Compute individual objective function scores
        # (1) Heading angle score: how well the trajectory points toward goal
        score_heading_angles = self.score_heading_angle(paths, goal_pose)
        
        # (2) Velocity score: maximize forward speed, slow down near goal
        score_vel = self.score_vel(velocities, paths, goal_pose)
        
        # (3) Obstacle score: maintain distance from obstacles
        score_obstacles = self.score_obstacles(paths, nearest_obs)

        # Normalize all scores to comparable range [0, 1]
        score_heading_angles = normalize(score_heading_angles)
        score_vel = normalize(score_vel)
        score_obstacles = normalize(score_obstacles)

        # Compute overall weighted score and select best trajectory
        opt_idx = np.argmax(np.sum(
            np.array([score_heading_angles, score_vel, score_obstacles])
            * np.array([[self.weight_angle, self.weight_vel, self.weight_obs]]).T,
            axis=0,
        ))

        try:
            return opt_idx
        except:
            raise Exception("Not possible to find an optimal path")

    def score_heading_angle(self, path, goal_pose):
        """
        Score trajectories based on final heading angle toward goal.
        Higher score means better alignment with goal direction.
        
        Args:
            path: Trajectory points
            goal_pose: Target goal position
            
        Returns:
            score_angle: Heading score for each trajectory
        """
        # Extract final position and heading from each trajectory
        last_x = path[:, -1, 0]
        last_y = path[:, -1, 1]
        last_th = path[:, -1, 2]

        # Calculate ideal heading angle to goal
        angle_to_goal = np.arctan2(goal_pose[1] - last_y, goal_pose[0] - last_x)

        # Compute score based on heading error
        score_angle = angle_to_goal - last_th
        score_angle = np.fabs(normalize_angle(score_angle))
        # Invert so 0 heading error gives maximum score
        score_angle = np.pi - score_angle

        return score_angle

    def score_vel(self, u, path, goal_pose):
        """
        Score trajectories based on velocity magnitude.
        Encourages faster motion, but reduces speed near goal.
        
        Args:
            u: Velocity commands
            path: Trajectory points
            goal_pose: Target goal position
            
        Returns:
            score: Velocity score for each trajectory
        """
        # Use linear velocity directly as base score
        vel = u[:, 0]
        
        # Calculate distance to goal at end of trajectory
        dist_to_goal = np.linalg.norm(path[:, -1, 0:2] - goal_pose, axis=-1)
        
        # Exponentially decrease score near goal to encourage deceleration
        score = vel + np.exp(-dist_to_goal / self.goal_dist_tol)
        return score

    def score_obstacles(self, path, obstacles):
        """
        Score trajectories based on distance to nearest obstacle.
        Penalizes trajectories that get too close to obstacles.
        
        Args:
            path: Trajectory points of shape (n_paths, sim_step, 2)
            obstacles: List of obstacle positions
            
        Returns:
            score_obstacle: Obstacle distance score for each trajectory
        """
        # Initialize all scores with maximum value (safe distance)
        score_obstacle = 2.0 * np.ones((path.shape[0]))

        # For each obstacle, compute minimum distance along each trajectory
        for obs in obstacles:
            # Calculate distance from all trajectory points to this obstacle
            dx = path[:, :, 0] - obs[0]
            dy = path[:, :, 1] - obs[1]
            dist = np.hypot(dx, dy)

            # Find minimum distance along each trajectory
            min_dist = np.min(dist, axis=-1)
            
            # Update scores with minimum distances (keep smallest values)
            score_obstacle[min_dist < score_obstacle] = min_dist[min_dist < score_obstacle]
        
            # Apply heavy penalty for collision trajectories
            score_obstacle[score_obstacle < self.robot.radius + self.collision_tol] = -100
               
        return score_obstacle
