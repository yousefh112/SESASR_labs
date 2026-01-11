import rclpy
from rclpy.node import Node
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
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
        self.dt = 1.0 / float(self.control_hz)   # Control loop period (seconds)

        # Robot dynamic constraints (reachable velocities/dynamic window)
        self.max_lin_vel = 0.22      # Maximum linear velocity (m/s)
        self.min_lin_vel = 0.0       # Minimum linear velocity (m/s)
        self.max_ang_vel = 2.0       # Maximum angular velocity (rad/s)
        self.max_lin_acc = 0.2       # Maximum linear acceleration (m/s^2)
        self.max_ang_acc = 2.0       # Maximum angular acceleration (rad/s^2)

        # DWA sampling parameters
        self.sim_time = 1.0          # Time horizon for simulating each candidate trajectory (seconds)
        self.time_granularity = 0.1  # Time step for trajectory simulation (seconds)
        self.v_samples = 10          # Number of linear velocity samples
        self.w_samples = 17          # Number of angular velocity samples (2D velocity search space)

        # LiDAR processing parameters
        self.lidar_max_range = 3.5   # Maximum LiDAR range to consider (meters)
        self.num_sectors = 26        # Number of angular sectors for LiDAR downsampling [12..30]
        self.scan_lock = Lock()      # Thread lock for scan data access
        self.current_scan_raw = None # Raw LiDAR ranges
        self.current_scan_sectors = None  # Downsampled sector distances (size: num_sectors)
        self.scan_angles = None             # Angles for each sector (center of each sector, radians)

        # Robot geometry and safety parameters
        self.robot_radius = 0.17     # Robot collision radius (meters, TurtleBot3 approximation)
        self.collision_tolerance = 0.17   # Additional safety margin (meters)
        self.collision_threshold = self.collision_tolerance # Total distance threshold for collision detection

        # DWA objective function weights
        self.alpha = 1.5  # Weight for heading alignment score
        self.beta = 1.2   # Weight for velocity score
        self.gamma = 0.9  # Weight for obstacle distance score
        self.delta = 2.0  # Weight for target following score

        # Target following parameters
        self.optimal_target_dist = 0.4 # Ideal distance to maintain from target (meters)
        self.slowing_distance = 0.25   # Distance threshold to start slowing down (meters)

        # Goal and state variables
        self.goal = None            # Target goal position as np.array([x, y])
        self.pose = None            # Current robot pose as np.array([x, y, yaw])
        self.vel = np.array([0.0, 0.0])  # Current velocity [linear_velocity, angular_velocity] from odometry
        self.control_step = 0       # Current control loop iteration counter
        self.max_control_steps = 2000 # Maximum control steps before timeout

        # Feedback and publishing parameters
        self.feedback_rate = 50     # Publish feedback every N control steps
        self.feedback_pub = self.create_publisher(Float32, '/dwa/goal_distance', 10)

        # ROS 2 Publishers and Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # Optional dynamic goal input (for dynamic targets from simulator)
        self.create_subscription(LandmarkArray, '/camera/landmarks', self.landmark_callback, 10)
        # Control loop timer (15 Hz)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('DWA Node initialized (running at {:.1f} Hz)'.format(self.control_hz))

    # ---------------------------
    # Subscription Callbacks
    # ---------------------------
    def odom_callback(self, msg: Odometry):
        """
        Update robot pose and velocity from odometry messages.
        Extracts position (x, y), orientation (yaw), and velocities.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.pose = np.array([x, y, yaw])
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        self.vel = np.array([vx, wz])

    def landmark_callback(self, msg):
        """
        Process AprilTag landmarks and convert to goal position.
        Transforms range and bearing measurements to global X-Y coordinates in odom frame.
        
        Args:
            msg: LandmarkArray message containing detected AprilTags
        """
        if self.pose is None or len(msg.landmarks) == 0:
            return
        
        # Extract range and bearing from first detected landmark
        tag = msg.landmarks[0] 
        r = tag.range       # Distance to landmark (meters)
        b = tag.bearing     # Bearing angle relative to robot (radians)
        
        # Current robot state
        robot_x, robot_y, robot_yaw = self.pose
        
        # Transform landmark position to global frame
        global_angle = robot_yaw + b
        goal_x = robot_x + r * math.cos(global_angle)
        goal_y = robot_y + r * math.sin(global_angle)
        
        # Update goal position
        self.goal = np.array([goal_x, goal_y])

    def scan_callback(self, msg: LaserScan):
        """
        Process raw LiDAR scan data.
        Handles NaN/Inf values, truncates to max range, and downsamples into sectors.
        Keeps minimum distance in each sector for obstacle detection.
        
        Args:
            msg: LaserScan message from LiDAR sensor
        """
        with self.scan_lock:
            ranges = np.array(msg.ranges, dtype=float)
            nan_mask = np.isnan(ranges)
            inf_mask = np.isposinf(ranges) | np.isneginf(ranges)
            
            if ranges.size == 0:
                return
            
            # Handle invalid values (NaN/Inf)
            ranges[nan_mask] = msg.range_max if msg.range_max is not None else 0.0
            ranges[inf_mask] = msg.range_max if msg.range_max is not None else self.lidar_max_range
            
            # Clamp negative and overly large ranges
            ranges[ranges < 0.0] = 0.0
            ranges[ranges > self.lidar_max_range] = self.lidar_max_range

            self.current_scan_raw = ranges

            # Downsample into angular sectors by taking minimum in each sector
            total = ranges.shape[0]
            N = max(12, min(30, self.num_sectors))
            chunk = int(total / N)
            if chunk < 1:
                chunk = 1
                N = total
            
            sectors = np.zeros(N)
            for i in range(N):
                start = i * chunk
                end = start + chunk if (i < N - 1) else total
                seg = ranges[start:end]
                # Set to max range if segment is empty
                if seg.size == 0:
                    sectors[i] = self.lidar_max_range
                else:
                    sectors[i] = np.min(seg)
            
            self.current_scan_sectors = sectors
            
            # Compute sector center angles (assuming TurtleBot3: -135..135 degrees)
            angles = np.linspace(-135.0, 135.0, N) * math.pi / 180.0
            self.scan_angles = angles

    # ---------------------------
    # Goal Management
    # ---------------------------
    def set_goal(self, goal_xy):
        """
        Set a new goal position and reset control step counter.
        
        Args:
            goal_xy: Target position as [x, y] in odom frame
        """
        self.goal = np.array(goal_xy, dtype=float)
        self.control_step = 0
        self.get_logger().info(f"New goal set: {self.goal}")

    # ---------------------------
    # Main Control Loop (Timer Callback)
    # ---------------------------
    def control_loop(self):
        """
        Main DWA control loop executed at configured frequency (15 Hz).
        Handles collision detection, DWA computation, and goal reaching logic.
        """
        # Wait until we have all required data
        if self.pose is None or self.current_scan_sectors is None or self.goal is None:
            return

        self.control_step += 1

        # Timeout check: stop if max control steps exceeded
        if self.control_step > self.max_control_steps:
            self.get_logger().warn('Timeout reached -> stopping')
            self.stop_robot()
            return

        # Publish distance feedback periodically
        if self.control_step % self.feedback_rate == 0:
            dist = np.linalg.norm(self.goal - self.pose[0:2])
            msg = Float32()
            msg.data = float(dist)
            self.feedback_pub.publish(msg)
            self.get_logger().info(f"[Feedback] distance to goal: {dist:.3f} m (step {self.control_step})")

        # Collision safety check: immediate stop if any obstacle is too close
        with self.scan_lock:
            min_range = float(np.min(self.current_scan_sectors)) if self.current_scan_sectors is not None else float('inf')
        if min_range < self.collision_threshold:
            self.get_logger().warn(f"Collision imminent (min_range={min_range:.3f} < threshold={self.collision_threshold:.3f}) -> STOP")
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
            self.get_logger().info("Goal reached ✔")
            self.stop_robot()
            # Reset goal to idle until new goal arrives
            self.goal = None

    # ---------------------------
    # DWA Core Algorithm
    # ---------------------------
    def compute_dwa_cmd(self):
        """
        Compute optimal velocity command using Dynamic Window Approach.
        
        Returns:
            (v_cmd, w_cmd): Optimal linear and angular velocities
        """
        # Step 1: Compute dynamic window bounds (reachable velocities)
        min_v, max_v, min_w, max_w = self.compute_dynamic_window(self.vel)

        # Step 2: Sample velocities uniformly in dynamic window
        v_values = np.linspace(min_v, max_v, self.v_samples)
        w_values = np.linspace(min_w, max_w, self.w_samples)

        best_score = -1e9
        best_cmd = (0.0, 0.0)

        # Precompute obstacle positions from current scan sectors (in odom frame)
        obst_points = self.scan_sectors_to_points()

        # Precompute current distance to goal
        dist_to_goal_now = np.linalg.norm(self.goal - self.pose[0:2])
        
        # Step 3: Evaluate each candidate velocity pair
        for v in v_values:
            for w in w_values:
                # Simulate trajectory for this velocity pair
                traj = self.simulate_trajectory(self.pose, v, w)
                if traj is None:
                    continue
                
                # Collision check: discard if trajectory passes through obstacle
                min_dist_traj = self.min_dist_to_obstacles(traj, obst_points)
                if min_dist_traj < self.robot_radius + 0.0:
                    continue

                # Evaluate multiple objective functions
                heading_score = self.score_heading(traj)
                obs_score = self.score_obstacle_distance(min_dist_traj)
                vel_score = self.score_velocity_adaptive(v, dist_to_goal_now)  # Slows down near goal
                target_score = self.score_target_following(traj)  # Maintains optimal distance to target
                
                # Compute weighted objective: maximize J
                J = (self.alpha * heading_score) + (self.beta * vel_score) + (self.gamma * obs_score) + (self.delta * target_score)

                # Track best trajectory
                if J > best_score:
                    best_score = J
                    best_cmd = (v, w)

        # If no valid trajectory found, stop (rare case)
        if best_score < -1e8:
            return 0.0, 0.0
        return best_cmd

    def score_velocity_adaptive(self, v, dist_to_goal):
        """
        Score velocity with adaptive behavior: slow down when approaching goal.
        
        Task 2: Implements velocity adaptation for close-range target following.
        
        Args:
            v: Candidate linear velocity (m/s)
            dist_to_goal: Current distance to goal (meters)
            
        Returns:
            score: Normalized score in [0, 1], higher is better
        """
        # Reduce ideal velocity as robot approaches goal
        if dist_to_goal < self.slowing_distance:
            # Scale down velocity proportionally to remaining distance
            v_ideal = self.max_lin_vel * (dist_to_goal / self.slowing_distance)
            v_ideal = max(0.05, v_ideal)  # Ensure minimum velocity
        else:
            v_ideal = self.max_lin_vel
            
        # Score based on how close candidate velocity matches ideal velocity
        diff = abs(v - v_ideal)
        score = 1.0 - (diff / self.max_lin_vel)
        return max(0.0, score)

    def score_target_following(self, traj):
        """
        Score trajectory based on final distance to target.
        
        Task 2: Evaluates if trajectory endpoint maintains optimal distance to dynamic target.
        Encourages robot to follow target at consistent distance.
        
        Args:
            traj: Simulated trajectory (n_steps x 3 array)
            
        Returns:
            score: Normalized score in [0, 1], higher is better
        """
        if traj is None or self.goal is None:
            return 0.0
            
        # Get final position from trajectory
        final_pose = traj[-1]
        final_xy = final_pose[0:2]
        
        # Compute distance from trajectory endpoint to goal
        d_end = np.linalg.norm(self.goal - final_xy)
        
        # Penalize deviation from optimal distance
        error = abs(d_end - self.optimal_target_dist)
        
        # Normalize error to [0, 1] score
        max_err = 2.0
        score = (max_err - error) / max_err
        return max(0.0, min(1.0, score))

        
    def compute_dynamic_window(self, current_vel):
        """
        Compute reachable velocity space given acceleration limits.
        
        The dynamic window is the intersection of:
        - Velocity space bounded by current velocity and acceleration limits
        - Hard limits on maximum velocity
        
        Args:
            current_vel: Current velocity [v_linear, v_angular]
            
        Returns:
            (min_v, max_v, min_w, max_w): Dynamic window bounds
        """
        v_curr, w_curr = float(current_vel[0]), float(current_vel[1])
        
        # Linear velocity window: limited by acceleration and max velocity
        min_v = max(self.min_lin_vel, v_curr - self.max_lin_acc * self.dt)
        max_v = min(self.max_lin_vel, v_curr + self.max_lin_acc * self.dt)
        
        # Angular velocity window: limited by acceleration and max velocity
        min_w = max(-self.max_ang_vel, w_curr - self.max_ang_acc * self.dt)
        max_w = min(self.max_ang_vel, w_curr + self.max_ang_acc * self.dt)
        
        return min_v, max_v, min_w, max_w

    def simulate_trajectory(self, start_pose, v, w):
        """
        Simulate robot trajectory forward in time.
        
        Uses differential drive kinematic model with constant velocity inputs.
        
        Args:
            start_pose: Initial pose [x, y, yaw]
            v: Linear velocity (m/s)
            w: Angular velocity (rad/s)
            
        Returns:
            traj: Trajectory as (n_steps x 3) array with [x, y, yaw] at each step
        """
        t_steps = int(max(1, round(self.sim_time / self.time_granularity)))
        traj = np.zeros((t_steps, 3))
        x, y, yaw = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
        dt_sim = self.time_granularity
        
        # Forward simulate using differential drive kinematics
        for i in range(t_steps):
            x += v * math.cos(yaw) * dt_sim
            y += v * math.sin(yaw) * dt_sim
            yaw += w * dt_sim
            yaw = self.normalize_angle(yaw)  # Keep angle in [-pi, pi]
            traj[i, :] = [x, y, yaw]
        
        return traj

    def scan_sectors_to_points(self):
        """
        Convert LiDAR sectors to obstacle point cloud in odom frame.
        
        Transforms sector distances and angles (relative to robot) to Cartesian
        coordinates in the global odometry frame.
        
        Returns:
            pts: Obstacle points as (N x 2) array with [x, y] coordinates
        """
        pts = []
        with self.scan_lock:
            sectors = self.current_scan_sectors
            angles = self.scan_angles
            pose = self.pose.copy() if self.pose is not None else None
        
        if sectors is None or angles is None or pose is None:
            return np.zeros((0, 2))
        
        rx, ry, ryaw = pose  # Robot position and heading
        
        # Transform each sector measurement to global coordinates
        for d, ang in zip(sectors, angles):
            # Compute global angle (robot heading + relative angle)
            gx = rx + d * math.cos(ryaw + ang)
            gy = ry + d * math.sin(ryaw + ang)
            pts.append([gx, gy])
        
        return np.array(pts)

    def min_dist_to_obstacles(self, traj, obst_pts):
        """
        Compute minimum distance between trajectory and all obstacles.
        
        Uses vectorized computation to efficiently find the closest obstacle
        point to any point on the trajectory.
        
        Args:
            traj: Trajectory as (T x 3) array
            obst_pts: Obstacle points as (N x 2) array
            
        Returns:
            min_dist: Minimum Euclidean distance from trajectory to obstacles
        """
        if traj is None or traj.shape[0] == 0:
            return 0.0
        if obst_pts is None or obst_pts.shape[0] == 0:
            return 1e3  # Large value if no obstacles
        
        # Vectorized distance computation
        px = traj[:, 0:2][:, None, :]  # (T, 1, 2)
        obs = obst_pts[None, :, :]     # (1, N, 2)
        diff = px - obs                # (T, N, 2)
        dists = np.linalg.norm(diff, axis=-1)  # (T, N)
        min_dist = float(np.min(dists))
        
        return min_dist

    # ---------------------------
    # Scoring Functions / Objective Components
    # ---------------------------
    def score_heading(self, traj):
        """
        Score trajectory based on heading alignment with goal.
        
        Evaluates how well the final trajectory heading points toward the goal.
        
        Args:
            traj: Simulated trajectory
            
        Returns:
            score: Normalized score in [0, 1], higher is better
        """
        if traj is None or traj.shape[0] == 0 or self.goal is None:
            return 0.0
        
        # Get final robot pose
        last = traj[-1]
        dx = self.goal[0] - last[0]
        dy = self.goal[1] - last[1]
        goal_angle = math.atan2(dy, dx)
        
        # Compute angular difference
        diff = abs(self.normalize_angle(goal_angle - last[2]))
        
        # Convert angle difference to score [0, 1] (1 = perfectly aligned)
        score = (math.pi - diff) / math.pi
        return max(0.0, min(1.0, score))

    def score_velocity(self, v):
        """
        Score velocity based on magnitude (prefers higher speeds).
        
        Args:
            v: Linear velocity (m/s)
            
        Returns:
            score: Normalized score in [0, 1]
        """
        score = float(v / (self.max_lin_vel + 1e-6))
        return max(0.0, min(1.0, score))

    def score_obstacle_distance(self, dist):
        """
        Score distance to nearest obstacle.
        
        Normalized by sensor maximum range. Higher distance = higher score.
        
        Args:
            dist: Minimum distance to obstacles (meters)
            
        Returns:
            score: Normalized score in [0, 1]
        """
        score = float(dist / (self.lidar_max_range + 1e-6))
        return max(0.0, min(1.0, score))

    # ---------------------------
    # Utility Functions
    # ---------------------------
    @staticmethod
    def quaternion_to_yaw(q):
        """
        Convert quaternion to yaw angle (rotation around Z-axis).
        
        Args:
            q: Quaternion (geometry_msgs/Quaternion with w, x, y, z)
            
        Returns:
            yaw: Angle in radians [-pi, pi]
        """
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def normalize_angle(a):
        """
        Normalize angle to [-pi, pi] range.
        
        Args:
            a: Angle in radians
            
        Returns:
            a_normalized: Angle in [-pi, pi]
        """
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def stop_robot(self):
        """
        Send zero velocity command to stop the robot immediately.
        """
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

def main(args=None):
    """
    Main entry point for DWA node.
    Initializes ROS 2, creates DWA_Node, and starts control loop.
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
