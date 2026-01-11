import rclpy
from rclpy.node import Node
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from threading import Lock
import time

class DWA_Node(Node):
    def __init__(self):
        super().__init__('dwa_node')

        # -------------------------
        # Parameters
        # -------------------------
        # Timing parameters
        self.control_hz = 15.0
        self.dt = 1.0 / float(self.control_hz)   # Control loop period in seconds

        # Robot dynamic constraints 
        # Reachable velocities/dynamic window based on robot capabilities
        self.max_lin_vel = 0.22      # Maximum linear velocity (m/s)
        self.min_lin_vel = 0.0       # Minimum linear velocity (m/s)
        self.max_ang_vel = 2.0       # Maximum angular velocity (rad/s)
        self.max_lin_acc = 0.2       # Maximum linear acceleration (m/s^2)
        self.max_ang_acc = 2.0       # Maximum angular acceleration (rad/s^2)

        # DWA (Dynamic Window Approach) sampling parameters
        self.sim_time = 1.0          # Time horizon for simulating each candidate trajectory (seconds)
        self.time_granularity = 0.1  # Time step for trajectory simulation (seconds)
        self.v_samples = 10          # Number of linear velocity samples to evaluate
        self.w_samples = 17          # Number of angular velocity samples to evaluate (2D velocity search space)

        # LiDAR processing parameters
        self.lidar_max_range = 3.5   # Maximum range to consider from LiDAR (meters); truncate beyond this
        self.num_sectors = 26        # Number of angular sectors to divide LiDAR scan into (12-30 recommended)
        self.scan_lock = Lock()      # Thread lock for safe LiDAR data access
        self.current_scan_raw = None # Raw LiDAR range readings
        self.current_scan_sectors = None  # Post-processed downsampled distances (size: num_sectors)
        self.scan_angles = None             # Center angles for each sector (in radians)

        # Robot geometry & collision safety
        self.robot_radius = 0.17     # Physical robot radius (meters, TurtleBot3 approximate)
        self.collision_tolerance = 0.17   # Safety margin beyond robot radius (meters)
        self.collision_threshold = self.collision_tolerance # Minimum safe distance to obstacles

        # DWA objective function weights
        # These weights balance different control objectives in the cost function
        self.alpha = 1.5   # Weight for heading alignment to goal (higher = prioritize facing goal)
        self.beta = 1.2    # Weight for velocity smoothness (higher = prefer constant speed)
        self.gamma = 0.9   # Weight for obstacle distance (higher = prioritize safety)
        self.delta = 2.0   # Weight for target following (higher = maintain distance to dynamic target)

        # Target following parameters (Task 2)
        self.optimal_target_dist = 1.2 # Desired distance to maintain from goal (meters)
        self.slowing_distance = 0.6    # Distance threshold at which robot begins slowing down (meters)

        # Goal and state tracking
        self.goal = None            # Goal position as np.array([x, y])
        self.pose = None            # Current robot pose as np.array([x, y, yaw])
        self.vel = np.array([0.0, 0.0])  # Current robot velocity: [linear_x, angular_z] from odometry
        self.control_step = 0       # Current control loop iteration counter
        self.max_control_steps = 2000 # Maximum steps before timeout (safety limit)

        # Feedback and diagnostics
        self.feedback_rate = 50     # Publish feedback every N control steps
        self.feedback_pub = self.create_publisher(Float32, '/dwa/goal_distance', 10)

        # ROS 2 Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # Optional dynamic goal input from simulator (provides /dynamic_goal_pose)
        self.create_subscription(Odometry, '/dynamic_goal_pose', self.dynamic_goal_callback, 10)
        # Note: Real robot would use: self.create_subscription(LandmarkArray, '/camera/landmarks', self.landmark_callback, 10)
        
        # Main control loop timer (runs at configured Hz)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('DWA Node initialized (running at {:.1f} Hz)'.format(self.control_hz))

    # ---------------------------
    # Callbacks
    # ---------------------------
    def odom_callback(self, msg: Odometry):
        """
        Callback: Update robot pose and velocity from odometry message.
        Called whenever new odometry data arrives.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.pose = np.array([x, y, yaw])
        
        # Extract current velocity from odometry
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        self.vel = np.array([vx, wz])

    def dynamic_goal_callback(self, msg: Odometry):
        """
        Callback: Set goal from dynamic goal topic (simulator support).
        If /dynamic_goal_pose exists, use it to update the target goal.
        """
        gx = msg.pose.pose.position.x
        gy = msg.pose.pose.position.y
        self.set_goal(np.array([gx, gy]))

    def scan_callback(self, msg: LaserScan):
        """
        Callback: Process LiDAR scan data.
        Steps:
          1. Handle NaN/Inf values
          2. Clamp ranges to valid bounds
          3. Downsample into angular sectors (keep minimum distance per sector)
          4. Compute sector center angles for obstacle point conversion
        """
        with self.scan_lock:
            ranges = np.array(msg.ranges, dtype=float)
            
            # Identify invalid values
            nan_mask = np.isnan(ranges)
            inf_mask = np.isposinf(ranges) | np.isneginf(ranges)
            
            if ranges.size == 0:
                return
            
            # Replace invalid values with sensor maximum range
            ranges[nan_mask] = msg.range_max if msg.range_max is not None else 0.0
            ranges[inf_mask] = msg.range_max if msg.range_max is not None else self.lidar_max_range
            
            # Clamp negative and out-of-range values
            ranges[ranges < 0.0] = 0.0
            ranges[ranges > self.lidar_max_range] = self.lidar_max_range

            self.current_scan_raw = ranges

            # Downsample scan into angular sectors by taking minimum distance per sector
            total = ranges.shape[0]
            N = max(12, min(30, self.num_sectors))  # Ensure sector count is in valid range
            chunk = int(total / N)
            if chunk < 1:
                chunk = 1
                N = total
                
            sectors = np.zeros(N)
            for i in range(N):
                start = i * chunk
                end = start + chunk if (i < N - 1) else total
                seg = ranges[start:end]
                # Use maximum range if segment is empty, otherwise use minimum distance
                if seg.size == 0:
                    sectors[i] = self.lidar_max_range
                else:
                    sectors[i] = np.min(seg)
            
            self.current_scan_sectors = sectors
            
            # Compute sector center angles in odom frame
            # TurtleBot3 LiDAR typically spans -135° to +135° (about 270° field of view)
            angles = np.linspace(-135.0, 135.0, N) * math.pi / 180.0
            self.scan_angles = angles

    # ---------------------------
    # Goal manager
    # ---------------------------
    def set_goal(self, goal_xy):
        """
        Set a new navigation goal and reset control step counter.
        Args:
            goal_xy: Target position as [x, y]
        """
        self.goal = np.array(goal_xy, dtype=float)
        self.control_step = 0
        self.get_logger().info(f"New goal set: {self.goal}")

    # ---------------------------
    # Main control loop (timer callback)
    # ---------------------------
    def control_loop(self):
        """
        Main DWA control loop executed at configured Hz.
        Steps:
          1. Check if all required data is available (pose, scan, goal)
          2. Check for timeout
          3. Publish diagnostic feedback
          4. Safety check: stop if obstacle too close
          5. Compute optimal DWA command
          6. Publish velocity command
          7. Check if goal is reached
        """
        # Wait until we have all necessary data
        if self.pose is None or self.current_scan_sectors is None or self.goal is None:
            return

        self.control_step += 1

        # Safety: timeout check
        if self.control_step > self.max_control_steps:
            self.get_logger().warn('Timeout reached -> stopping robot')
            self.stop_robot()
            return

        # Publish diagnostic feedback at specified rate
        if self.control_step % self.feedback_rate == 0:
            dist = np.linalg.norm(self.goal - self.pose[0:2])
            msg = Float32()
            msg.data = float(dist)
            self.feedback_pub.publish(msg)
            self.get_logger().info(f"[Feedback] distance to goal: {dist:.3f} m (step {self.control_step})")

        # Immediate safety stop: check if any sector is below collision threshold
        with self.scan_lock:
            min_range = float(np.min(self.current_scan_sectors)) if self.current_scan_sectors is not None else float('inf')
        
        if min_range < self.collision_threshold:
            self.get_logger().warn(f"Collision imminent (min_range={min_range:.3f} < threshold={self.collision_threshold:.3f}) -> STOP")
            self.stop_robot()
            return

        # Compute optimal velocity command using DWA
        v_cmd, w_cmd = self.compute_dwa_cmd()

        # Publish velocity command to robot
        twist = Twist()
        twist.linear.x = float(v_cmd)
        twist.angular.z = float(w_cmd)
        self.cmd_pub.publish(twist)

        # Check if goal has been reached (within tolerance)
        dist_to_goal = np.linalg.norm(self.goal - self.pose[0:2])
        if dist_to_goal < 0.15:
            self.get_logger().info("Goal reached ✔")
            self.stop_robot()
            # Reset goal so loop idles until new goal arrives
            self.goal = None

    # ---------------------------
    # DWA core algorithm
    # ---------------------------
    def compute_dwa_cmd(self):
        """
        Compute optimal (v, w) command using Dynamic Window Approach.
        Algorithm:
          1. Compute dynamic window based on current velocity and acceleration limits
          2. Sample candidate velocities uniformly in the window
          3. For each candidate, simulate trajectory and evaluate cost
          4. Return velocity pair with highest cost
        Returns:
            (v, w): Linear and angular velocity command
        """
        # Step 1: Compute reachable velocity bounds (dynamic window)
        min_v, max_v, min_w, max_w = self.compute_dynamic_window(self.vel)

        # Step 2: Sample velocities uniformly across the window
        v_values = np.linspace(min_v, max_v, self.v_samples)
        w_values = np.linspace(min_w, max_w, self.w_samples)

        best_score = -1e9
        best_cmd = (0.0, 0.0)

        # Precompute obstacle points from current LiDAR scan
        obst_points = self.scan_sectors_to_points()

        # Task 2: Precompute distance to target for adaptive velocity scoring
        dist_to_goal_now = np.linalg.norm(self.goal - self.pose[0:2])
        
        # Step 3: Evaluate each candidate velocity pair
        for v in v_values:
            for w in w_values:
                # Simulate trajectory for this (v, w) pair
                traj = self.simulate_trajectory(self.pose, v, w)
                if traj is None:
                    continue
                
                # Check collision: if trajectory passes too close to obstacle, discard
                min_dist_traj = self.min_dist_to_obstacles(traj, obst_points)
                if min_dist_traj < self.robot_radius + 0.0:
                    continue  # Collision trajectory; skip this candidate

                # Evaluate objective function components (each in [0, 1])
                heading_score = self.score_heading(traj)        # Goal direction alignment
                obs_score = self.score_obstacle_distance(min_dist_traj)  # Safety margin
                vel_score = self.score_velocity_adaptive(v, dist_to_goal_now)  # Speed preference (adaptive)
                target_score = self.score_target_following(traj)  # Maintain distance to target
                
                # Weighted sum of objective components
                J = (self.alpha * heading_score) + (self.beta * vel_score) + (self.gamma * obs_score) + (self.delta * target_score)

                # Track best candidate
                if J > best_score:
                    best_score = J
                    best_cmd = (v, w)

        # Step 4: Return best command or zero if no valid candidate found
        if best_score < -1e8:
            return 0.0, 0.0
        return best_cmd

    def score_velocity_adaptive(self, v, dist_to_goal):
        """
        Score linear velocity with adaptive behavior based on distance to goal.
        Task 2: Slow down when approaching target.
        
        Logic:
          - If close to goal (< slowing_distance): ramp down speed proportionally
          - Otherwise: prefer maximum velocity
          - Score: how well does v match ideal velocity (1.0 = perfect match)
        
        Args:
            v: Candidate linear velocity (m/s)
            dist_to_goal: Current distance to goal (meters)
        Returns:
            score: Value in [0, 1], where 1 is best
        """
        # Compute ideal velocity based on distance to goal
        if dist_to_goal < self.slowing_distance:
            # Ramp down speed as we approach: ideal_v = max_v * (dist / slowing_dist)
            v_ideal = self.max_lin_vel * (dist_to_goal / self.slowing_distance)
            v_ideal = max(0.05, v_ideal)  # Ensure minimum non-zero speed
        else:
            # Far from goal: prefer maximum velocity
            v_ideal = self.max_lin_vel
        
        # Score based on deviation from ideal velocity
        diff = abs(v - v_ideal)
        score = 1.0 - (diff / self.max_lin_vel)
        return max(0.0, score)

    def score_target_following(self, traj):
        """
        Score how well the trajectory maintains desired distance to dynamic target.
        Task 2: Evaluate if trajectory endpoint reaches optimal_target_dist.
        
        Logic:
          - Compute distance from trajectory endpoint to goal
          - Compare against optimal_target_dist
          - Score: 1.0 = perfect distance, 0.0 = too far off
        
        Args:
            traj: Simulated trajectory (array of [x, y, yaw] poses)
        Returns:
            score: Value in [0, 1], where 1 is best
        """
        if traj is None or self.goal is None:
            return 0.0
        
        # Get final position of trajectory
        final_pose = traj[-1]
        final_xy = final_pose[0:2]
        
        # Compute distance from trajectory end to goal
        d_end = np.linalg.norm(self.goal - final_xy)
        
        # Compute error from optimal distance
        error = abs(d_end - self.optimal_target_dist)
        
        # Normalize error to [0, 1] score
        max_err = 2.0  # Maximum acceptable error (meters)
        score = (max_err - error) / max_err
        return max(0.0, min(1.0, score))

        
    def compute_dynamic_window(self, current_vel):
        """
        Compute reachable velocity window based on current velocity and acceleration limits.
        
        The dynamic window constrains achievable velocities in one control step:
          Vd = [v_curr - a_max*dt, v_curr + a_max*dt] ∩ [v_min, v_max]
        
        This ensures only kinematically feasible accelerations are considered.
        
        Args:
            current_vel: Current [linear_vel, angular_vel]
        Returns:
            (min_v, max_v, min_w, max_w): Velocity window bounds
        """
        v_curr, w_curr = float(current_vel[0]), float(current_vel[1])
        
        # Linear velocity window: apply acceleration limits and robot limits
        min_v = max(self.min_lin_vel, v_curr - self.max_lin_acc * self.dt)
        max_v = min(self.max_lin_vel, v_curr + self.max_lin_acc * self.dt)
        
        # Angular velocity window: apply acceleration limits and robot limits
        min_w = max(-self.max_ang_vel, w_curr - self.max_ang_acc * self.dt)
        max_w = min(self.max_ang_vel, w_curr + self.max_ang_acc * self.dt)
        
        return min_v, max_v, min_w, max_w

    def simulate_trajectory(self, start_pose, v, w):
        """
        Simulate robot trajectory under constant (v, w) control for prediction horizon.
        
        Uses simple differential drive kinematics:
          x(t+dt) = x(t) + v*cos(yaw)*dt
          y(t+dt) = y(t) + v*sin(yaw)*dt
          yaw(t+dt) = yaw(t) + w*dt
        
        Args:
            start_pose: Initial robot state [x, y, yaw]
            v: Linear velocity (m/s)
            w: Angular velocity (rad/s)
        Returns:
            traj: Simulated trajectory as array of shape (n_steps, 3) containing [x, y, yaw]
        """
        # Calculate number of simulation steps based on time horizon
        t_steps = int(max(1, round(self.sim_time / self.time_granularity)))
        traj = np.zeros((t_steps, 3))
        
        # Initialize state
        x, y, yaw = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
        dt_sim = self.time_granularity
        
        # Simulate trajectory step by step
        for i in range(t_steps):
            # Update position using differential drive kinematics
            x += v * math.cos(yaw) * dt_sim
            y += v * math.sin(yaw) * dt_sim
            yaw += w * dt_sim
            yaw = self.normalize_angle(yaw)  # Keep angle in [-π, π]
            traj[i, :] = [x, y, yaw]
        
        return traj

    def scan_sectors_to_points(self):
        """
        Convert LiDAR sector readings to obstacle points in odom frame.
        
        Each LiDAR sector is defined by (distance, angle_relative_to_robot).
        Transform to global odom frame using current robot pose.
        
        Returns:
            obst_pts: Array of shape (n_sectors, 2) with [x, y] coordinates in odom frame
        """
        pts = []
        with self.scan_lock:
            sectors = self.current_scan_sectors
            angles = self.scan_angles
            pose = self.pose.copy() if self.pose is not None else None
        
        # Return empty array if data not yet available
        if sectors is None or angles is None or pose is None:
            return np.zeros((0, 2))
        
        # Extract robot state
        rx, ry, ryaw = pose
        
        # Convert each sector to global coordinates
        for d, ang in zip(sectors, angles):
            # Angle is relative to robot heading; transform to odom frame
            # Global angle = robot_yaw + sector_angle
            gx = rx + d * math.cos(ryaw + ang)
            gy = ry + d * math.sin(ryaw + ang)
            pts.append([gx, gy])
        
        return np.array(pts)

    def min_dist_to_obstacles(self, traj, obst_pts):
        """
        Compute minimum Euclidean distance between simulated trajectory and obstacles.
        
        Uses vectorized computation for efficiency:
          - traj: sequence of robot poses (T, 2)
          - obst_pts: set of obstacle points (N, 2)
          - Output: scalar minimum distance across all (pose, obstacle) pairs
        
        Args:
            traj: Trajectory array of shape (n_steps, 3); uses only (x, y) columns
            obst_pts: Obstacle points array of shape (n_obstacles, 2)
        Returns:
            min_dist: Scalar minimum distance (meters)
        """
        # Return high distance if trajectory is empty or no obstacles
        if traj is None or traj.shape[0] == 0:
            return 0.0
        if obst_pts is None or obst_pts.shape[0] == 0:
            return 1e3
        
        # Vectorized distance computation
        px = traj[:, 0:2][:, None, :]  # Shape (T, 1, 2) - trajectory positions
        obs = obst_pts[None, :, :]     # Shape (1, N, 2) - obstacle positions
        diff = px - obs                # Shape (T, N, 2) - all position differences
        dists = np.linalg.norm(diff, axis=-1)  # Shape (T, N) - all distances
        min_dist = float(np.min(dists))  # Scalar minimum
        
        return min_dist

    # ---------------------------
    # Scoring / normalization helpers
    # ---------------------------
    def score_heading(self, traj):
        """
        Score how well trajectory is aligned with goal direction.
        
        Compares final pose heading with direction to goal.
        Scoring: 1.0 = perfectly aligned, 0.0 = opposite direction
        
        Args:
            traj: Simulated trajectory
        Returns:
            score: Value in [0, 1], where 1 is best
        """
        if traj is None or traj.shape[0] == 0 or self.goal is None:
            return 0.0
        
        # Get final pose from trajectory
        last = traj[-1]
        
        # Compute angle to goal from final position
        dx = self.goal[0] - last[0]
        dy = self.goal[1] - last[1]
        goal_angle = math.atan2(dy, dx)
        
        # Compute angle difference (normalized to [-π, π])
        diff = abs(self.normalize_angle(goal_angle - last[2]))
        
        # Convert angle difference to score in [0, 1]
        # diff=0 → score=1 (perfect), diff=π → score=0 (opposite)
        score = (math.pi - diff) / math.pi
        return max(0.0, min(1.0, score))

    def score_velocity(self, v):
        """
        Score linear velocity preference (basic, unused in adaptive version).
        Prefers higher velocities up to maximum.
        
        Args:
            v: Linear velocity (m/s)
        Returns:
            score: Value in [0, 1], where 1 is best
        """
        score = float(v / (self.max_lin_vel + 1e-6))
        return max(0.0, min(1.0, score))

    def score_obstacle_distance(self, dist):
        """
        Score safety based on distance to nearest obstacle.
        Larger distances are preferred.
        
        Args:
            dist: Minimum distance to obstacle (meters)
        Returns:
            score: Value in [0, 1], where 1 is best (far from obstacles)
        """
        # Normalize by sensor range; clamp to [0, 1]
        score = float(dist / (self.lidar_max_range + 1e-6))
        return max(0.0, min(1.0, score))

    # ---------------------------
    # Utility functions
    # ---------------------------
    @staticmethod
    def quaternion_to_yaw(q):
        """
        Convert quaternion orientation to yaw angle (rotation around Z-axis).
        
        Standard ROS quaternion to Euler conversion for planar yaw.
        Args:
            q: geometry_msgs/Quaternion with fields (x, y, z, w)
        Returns:
            yaw: Angle in radians in [-π, π]
        """
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def normalize_angle(a):
        """
        Normalize angle to [-π, π] range.
        
        Args:
            a: Angle in radians
        Returns:
            Normalized angle in [-π, π]
        """
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def stop_robot(self):
        """
        Immediately stop robot by publishing zero velocity command.
        """
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

def main(args=None):
    """
    Main entry point: Initialize ROS 2 node and run control loop.
    """
    rclpy.init(args=args)
    node = DWA_Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
