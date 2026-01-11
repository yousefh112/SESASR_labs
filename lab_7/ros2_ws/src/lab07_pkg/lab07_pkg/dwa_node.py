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
        # Configuration Parameters
        # -------------------------
        # Control timing
        self.control_hz = 15.0
        self.dt = 1.0 / float(self.control_hz)   # Control loop period in seconds

        # Robot dynamic constraints and motion limits
        # These define the reachable velocities within the dynamic window
        self.max_lin_vel = 0.22      # Maximum linear velocity in m/s
        self.min_lin_vel = 0.0       # Minimum linear velocity in m/s
        self.max_ang_vel = 2.0       # Maximum angular velocity in rad/s
        self.max_lin_acc = 0.2       # Maximum linear acceleration in m/s^2
        self.max_ang_acc = 2.0       # Maximum angular acceleration in rad/s^2

        # Dynamic Window Approach (DWA) sampling parameters
        self.sim_time = 1.0          # Simulation horizon: seconds to simulate each candidate trajectory
        self.time_granularity = 0.1  # Time step for trajectory rollout simulation in seconds
        self.v_samples = 10          # Number of linear velocity samples in dynamic window
        self.w_samples = 17          # Number of angular velocity samples in dynamic window

        # LiDAR sensor processing configuration
        self.lidar_max_range = 3.5   # Maximum range to truncate LiDAR readings in meters
        self.num_sectors = 26        # Number of angular sectors for downsampling scan (typical: 12-30)
        self.scan_lock = Lock()      # Thread-safe access to scan data
        self.current_scan_raw = None # Raw LiDAR range array
        self.current_scan_sectors = None  # Downsampled distance measurements (size: num_sectors)
        self.scan_angles = None             # Center angles for each sector in robot frame

        # Robot physical properties and collision detection
        self.robot_radius = 0.17     # Robot footprint radius in meters (TurtleBot3 approx)
        self.collision_tolerance = 0.17   # Safety buffer distance in meters
        self.collision_threshold = self.collision_tolerance # Minimum safe distance to obstacles

        # DWA objective function weights
        # These balance different goals: heading alignment, forward progress, and obstacle avoidance
        self.alpha = 1.5  # Weight for heading alignment score (goal direction)
        self.beta = 1.2   # Weight for velocity score (prefer forward motion)
        self.gamma = 0.9  # Weight for obstacle distance score (prefer clearance)

        # Goal and robot state tracking
        self.goal = None            # Target position as np.array([x, y])
        self.pose = None            # Current robot pose as np.array([x, y, yaw])
        self.vel = np.array([0.0, 0.0])  # Current velocity [linear_m/s, angular_rad/s] from odometry
        self.control_step = 0       # Counter for control loop iterations
        self.max_control_steps = 2000  # Maximum iterations before timeout

        # Feedback and monitoring
        self.feedback_rate = 50     # Publish distance feedback every N control steps
        self.feedback_pub = self.create_publisher(Float32, '/dwa/goal_distance', 10)

        # ROS 2 publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # Optional dynamic goal input from simulator
        self.create_subscription(Odometry, '/dynamic_goal_pose', self.dynamic_goal_callback, 10)
        # Main control loop timer
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('DWA Node initialized (running at {:.1f} Hz)'.format(self.control_hz))

    # ---------------------------
    # ROS 2 Message Callbacks
    # ---------------------------
    def odom_callback(self, msg: Odometry):
        """Extract and update robot pose and velocity from odometry message."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.pose = np.array([x, y, yaw])
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        self.vel = np.array([vx, wz])

    def dynamic_goal_callback(self, msg: Odometry):
        """Update goal position from dynamic goal pose message (simulator integration)."""
        gx = msg.pose.pose.position.x
        gy = msg.pose.pose.position.y
        self.set_goal(np.array([gx, gy]))

    def scan_callback(self, msg: LaserScan):
        """
        Process LiDAR scan data:
        - Handle invalid readings (NaN, Inf)
        - Truncate to maximum range
        - Downsample into angular sectors using minimum distance per sector
        """
        with self.scan_lock:
            ranges = np.array(msg.ranges, dtype=float)
            nan_mask = np.isnan(ranges)
            inf_mask = np.isposinf(ranges) | np.isneginf(ranges)
            
            if ranges.size == 0:
                return
            
            # Replace invalid values with maximum sensor range
            ranges[nan_mask] = msg.range_max if msg.range_max is not None else 0.0
            ranges[inf_mask] = msg.range_max if msg.range_max is not None else self.lidar_max_range
            
            # Clamp negative and overly long ranges
            ranges[ranges < 0.0] = 0.0
            ranges[ranges > self.lidar_max_range] = self.lidar_max_range

            self.current_scan_raw = ranges

            # Downsample into angular sectors
            total = ranges.shape[0]
            N = max(12, min(30, self.num_sectors))
            chunk = int(total / N)
            if chunk < 1:
                chunk = 1
                N = total
            
            # Keep minimum distance reading per sector
            sectors = np.zeros(N)
            for i in range(N):
                start = i * chunk
                end = start + chunk if (i < N - 1) else total
                seg = ranges[start:end]
                sectors[i] = np.min(seg) if seg.size > 0 else self.lidar_max_range
            
            self.current_scan_sectors = sectors
            
            # Compute sector center angles (approximately -135° to +135° for TurtleBot3)
            angles = np.linspace(-135.0, 135.0, N) * math.pi / 180.0
            self.scan_angles = angles

    # ---------------------------
    # Goal Management
    # ---------------------------
    def set_goal(self, goal_xy):
        """Set a new navigation goal and reset control step counter."""
        self.goal = np.array(goal_xy, dtype=float)
        self.control_step = 0
        self.get_logger().info(f"New goal set: {self.goal}")

    # ---------------------------
    # Main Control Loop
    # ---------------------------
    def control_loop(self):
        """
        Main DWA control loop called at configured frequency (15 Hz).
        Computes optimal velocity commands and publishes them.
        """
        # Wait until all required data is available
        if self.pose is None or self.current_scan_sectors is None or self.goal is None:
            return

        self.control_step += 1

        # Check for timeout
        if self.control_step > self.max_control_steps:
            self.get_logger().warn('Control timeout reached -> stopping robot')
            self.stop_robot()
            return

        # Publish distance-to-goal feedback at regular intervals
        if self.control_step % self.feedback_rate == 0:
            dist = np.linalg.norm(self.goal - self.pose[0:2])
            msg = Float32()
            msg.data = float(dist)
            self.feedback_pub.publish(msg)
            self.get_logger().info(f"[Feedback] Distance to goal: {dist:.3f} m (step {self.control_step})")

        # Safety check: emergency stop if obstacle too close
        with self.scan_lock:
            min_range = float(np.min(self.current_scan_sectors)) if self.current_scan_sectors is not None else float('inf')
        
        if min_range < self.collision_threshold:
            self.get_logger().warn(f"Collision imminent (min_range={min_range:.3f} m < threshold={self.collision_threshold:.3f} m) -> EMERGENCY STOP")
            self.stop_robot()
            return

        # Compute optimal velocity command using DWA
        v_cmd, w_cmd = self.compute_dwa_cmd()

        # Publish velocity command
        twist = Twist()
        twist.linear.x = float(v_cmd)
        twist.angular.z = float(w_cmd)
        self.cmd_pub.publish(twist)

        # Check if goal has been reached
        dist_to_goal = np.linalg.norm(self.goal - self.pose[0:2])
        if dist_to_goal < 0.15:
            self.get_logger().info("Goal reached! ✔")
            self.stop_robot()
            self.goal = None  # Reset to idle state
            return

    # ---------------------------
    # DWA Core Algorithm
    # ---------------------------
    def compute_dwa_cmd(self):
        """
        Compute optimal velocity command using Dynamic Window Approach:
        1. Calculate reachable velocity window from current state and acceleration limits
        2. Sample candidate velocities uniformly across window
        3. Simulate trajectory for each candidate
        4. Score trajectories on heading alignment, velocity, and obstacle clearance
        5. Return best-scoring velocity pair
        """
        # Step 1: Compute dynamic window based on current velocity and acceleration limits
        min_v, max_v, min_w, max_w = self.compute_dynamic_window(self.vel)

        # Step 2: Sample velocity grid within dynamic window
        v_values = np.linspace(min_v, max_v, self.v_samples)
        w_values = np.linspace(min_w, max_w, self.w_samples)

        best_score = -1e9
        best_cmd = (0.0, 0.0)

        # Precompute obstacle positions from current LiDAR scan
        obst_points = self.scan_sectors_to_points()

        # Step 3-5: Evaluate all velocity candidates
        for v in v_values:
            for w in w_values:
                # Simulate trajectory with constant (v, w)
                traj = self.simulate_trajectory(self.pose, v, w)
                if traj is None:
                    continue
                
                # Check collision: discard trajectories that collide with obstacles
                min_dist_traj = self.min_dist_to_obstacles(traj, obst_points)
                if min_dist_traj < self.robot_radius:
                    continue

                # Compute normalized objective function components
                heading_score = self.score_heading(traj)
                vel_score = self.score_velocity(v)
                obs_score = self.score_obstacle_distance(min_dist_traj)

                # Weighted sum of normalized scores
                J = (self.alpha * heading_score) + (self.beta * vel_score) + (self.gamma * obs_score)

                if J > best_score:
                    best_score = J
                    best_cmd = (v, w)

        # Return best command or zero if no valid candidates found
        if best_score < -1e8:
            return 0.0, 0.0
        return best_cmd

    def compute_dynamic_window(self, current_vel):
        """
        Calculate reachable velocity window considering:
        - Current velocity
        - Maximum acceleration limits
        - Robot velocity constraints
        
        Returns: (min_v, max_v, min_w, max_w)
        """
        v_curr, w_curr = float(current_vel[0]), float(current_vel[1])
        
        # Linear velocity window
        min_v = max(self.min_lin_vel, v_curr - self.max_lin_acc * self.dt)
        max_v = min(self.max_lin_vel, v_curr + self.max_lin_acc * self.dt)
        
        # Angular velocity window
        min_w = max(-self.max_ang_vel, w_curr - self.max_ang_acc * self.dt)
        max_w = min(self.max_ang_vel, w_curr + self.max_ang_acc * self.dt)
        
        return min_v, max_v, min_w, max_w

    def simulate_trajectory(self, start_pose, v, w):
        """
        Simulate robot trajectory over sim_time seconds using constant velocity (v, w).
        Uses simple differential drive kinematics.
        
        Args:
            start_pose: np.array([x, y, yaw]) - initial pose
            v: linear velocity in m/s
            w: angular velocity in rad/s
            
        Returns:
            traj: np.array shape (n_steps, 3) with [x, y, yaw] at each step
        """
        t_steps = int(max(1, round(self.sim_time / self.time_granularity)))
        traj = np.zeros((t_steps, 3))
        x, y, yaw = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
        dt_sim = self.time_granularity
        
        for i in range(t_steps):
            # Differential drive kinematics: update pose with constant (v, w)
            x += v * math.cos(yaw) * dt_sim
            y += v * math.sin(yaw) * dt_sim
            yaw += w * dt_sim
            yaw = self.normalize_angle(yaw)
            traj[i, :] = [x, y, yaw]
        
        return traj

    def scan_sectors_to_points(self):
        """
        Convert LiDAR scan sectors (distance + relative angle) to obstacle points in odometry frame.
        Each sector reading is converted to a point using current robot pose.
        
        Returns:
            np.array: (N, 2) array of [x, y] obstacle coordinates in odom frame
        """
        pts = []
        with self.scan_lock:
            sectors = self.current_scan_sectors
            angles = self.scan_angles
            pose = self.pose.copy() if self.pose is not None else None
        
        if sectors is None or angles is None or pose is None:
            return np.zeros((0, 2))
        
        rx, ry, ryaw = pose
        # Transform each sector reading from robot frame to odom frame
        for d, ang in zip(sectors, angles):
            # ang is relative to robot heading; transform using current yaw
            gx = rx + d * math.cos(ryaw + ang)
            gy = ry + d * math.sin(ryaw + ang)
            pts.append([gx, gy])
        
        return np.array(pts)

    def min_dist_to_obstacles(self, traj, obst_pts):
        """
        Compute minimum Euclidean distance from simulated trajectory to any obstacle point.
        
        Args:
            traj: trajectory array shape (T, 3) with [x, y, yaw]
            obst_pts: obstacle points array shape (N, 2) with [x, y]
            
        Returns:
            float: minimum distance in meters
        """
        if traj is None or traj.shape[0] == 0:
            return 0.0
        if obst_pts is None or obst_pts.shape[0] == 0:
            return 1e3
        
        # Vectorized distance computation
        px = traj[:, 0:2][:, None, :]  # Shape: (T, 1, 2)
        obs = obst_pts[None, :, :]     # Shape: (1, N, 2)
        diff = px - obs                # Shape: (T, N, 2)
        dists = np.linalg.norm(diff, axis=-1)  # Shape: (T, N)
        min_dist = float(np.min(dists))
        
        return min_dist

    # ---------------------------
    # Trajectory Scoring Functions
    # ---------------------------
    def score_heading(self, traj):
        """
        Score heading alignment: prefer trajectories that point toward goal.
        Compares final trajectory heading with direction to goal.
        
        Returns: score in [0, 1] where 1.0 = perfectly aligned
        """
        if traj is None or traj.shape[0] == 0 or self.goal is None:
            return 0.0
        
        last = traj[-1]
        dx = self.goal[0] - last[0]
        dy = self.goal[1] - last[1]
        goal_angle = math.atan2(dy, dx)
        
        # Compute angular difference in range [-π, π]
        diff = abs(self.normalize_angle(goal_angle - last[2]))
        
        # Convert angle error to score (π difference = 0, 0 difference = 1)
        score = (math.pi - diff) / math.pi
        return max(0.0, min(1.0, score))

    def score_velocity(self, v):
        """
        Score linear velocity: prefer higher forward velocity.
        Normalized by maximum achievable velocity.
        
        Returns: score in [0, 1] where 1.0 = max velocity
        """
        score = float(v / (self.max_lin_vel + 1e-6))
        return max(0.0, min(1.0, score))

    def score_obstacle_distance(self, dist):
        """
        Score obstacle clearance: prefer larger distances from obstacles.
        Normalized by sensor maximum range.
        
        Args:
            dist: minimum distance to obstacles in meters
            
        Returns: score in [0, 1] where 1.0 = max sensor range
        """
        score = float(dist / (self.lidar_max_range + 1e-6))
        return max(0.0, min(1.0, score))

    # ---------------------------
    # Utility Functions
    # ---------------------------
    @staticmethod
    def quaternion_to_yaw(q):
        """Convert quaternion to Euler yaw angle (roll and pitch assumed zero)."""
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def normalize_angle(a):
        """Normalize angle to range [-π, π] radians."""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def stop_robot(self):
        """Send zero velocity command to stop the robot."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

def main(args=None):
    """Initialize ROS 2 node and run DWA controller."""
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
