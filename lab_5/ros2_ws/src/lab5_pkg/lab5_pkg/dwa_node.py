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
        # Timing
        self.control_hz = 15.0
        self.dt = 1.0 / float(self.control_hz)   # control loop period

        # Robot dynamic constraints
        ######reachable velocities/dynamic window
        self.max_lin_vel = 0.20      # m/s(0.22)
        self.min_lin_vel = 0.0
        self.max_ang_vel = 1.0      # rad/s(2.84)
        self.max_lin_acc = 0.2       # m/s^2
        self.max_ang_acc = 2.0       # rad/s^2

        # DWA sampling params
        self.sim_time = 1.0          # seconds to simulate each candidate trajectory
        self.time_granularity = 0.1  # dt for simulating rollout
        self.v_samples = 10    #6
        self.w_samples = 17   # 2D velocity search space  12

        # LiDAR processing
        self.lidar_max_range = 3.5   # meters (truncate)  
        self.num_sectors = 26        # choose between [12..30]
        self.scan_lock = Lock()
        self.current_scan_raw = None
        self.current_scan_sectors = None  # post-processed downsampled distances (size num_sectors)
        self.scan_angles = None             # angles for sectors (center of each sector)

        # Robot geometry & safety
        self.robot_radius = 0.17     # m (TurtleBot3 approx)
        self.collision_tolerance = 0.15   # m (should be 0.15 - 0.25)
        self.collision_threshold = self.collision_tolerance #self.robot_radius + self.collision_tolerance

        # DWA weights (original objective)
        self.alpha = 1.0  # heading weight
        self.beta = 0.5   # velocity weight
        self.gamma = 1.0  # obstacle distance weight


        # Goal & state
        self.goal = None            # np.array([x,y])
        self.pose = None            # np.array([x,y,yaw])
        self.vel = np.array([0.0, 0.0])  # linear, angular current velocity (from odom)
        self.control_step = 0
        self.max_control_steps = 2000

        # Feedback
        self.feedback_rate = 50
        self.feedback_pub = self.create_publisher(Float32, '/dwa/goal_distance', 10)

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # Optional dynamic goal input (simulator provides /dynamic_goal_pose)
        self.create_subscription(Odometry, '/dynamic_goal_pose', self.dynamic_goal_callback, 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('DWA Node initialized (running at {:.1f} Hz)'.format(self.control_hz))

    # ---------------------------
    # Callbacks
    # ---------------------------
    def odom_callback(self, msg: Odometry):
        # Update robot pose and velocity from odom
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.pose = np.array([x, y, yaw])
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        self.vel = np.array([vx, wz])

    def dynamic_goal_callback(self, msg: Odometry):
        # If /dynamic_goal_pose exists (simulator), set goal from it
        gx = msg.pose.pose.position.x
        gy = msg.pose.pose.position.y
        self.set_goal(np.array([gx, gy]))

    def scan_callback(self, msg: LaserScan):
        # deal with LIDAR callback
        # Raw ranges -> process: NaN/Inf handling, truncation to lidar_max_range,
        # downsampling into sectors and keeping min in each sector.
        with self.scan_lock:
            ranges = np.array(msg.ranges, dtype=float)
            nan_mask = np.isnan(ranges)
            inf_mask = np.isposinf(ranges) | np.isneginf(ranges)
            if ranges.size == 0:
                return
            # assign according to PDF hint
            ranges[nan_mask] = msg.range_max if msg.range_max is not None else 0.0
            ranges[inf_mask] = msg.range_max if msg.range_max is not None else self.lidar_max_range
            # clamp negatives and long ranges
            ranges[ranges < 0.0] = 0.0
            ranges[ranges > self.lidar_max_range] = self.lidar_max_range

            self.current_scan_raw = ranges

            # Downsample into sectors
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
                # if empty segment, set to max
                if seg.size == 0:
                    sectors[i] = self.lidar_max_range
                else:
                    sectors[i] = np.min(seg)
            self.current_scan_sectors = sectors
            # Compute sector center angles (LaserScan angle_min/angle_max, approximate as -pi/2..pi/2 or -135..135)
            # We choose -135..135 deg as common for TurtleBot3
            angles = np.linspace(-135.0, 135.0, N) * math.pi / 180.0
            self.scan_angles = angles

    # ---------------------------
    # Goal manager
    # ---------------------------
    def set_goal(self, goal_xy):
        self.goal = np.array(goal_xy, dtype=float)
        self.control_step = 0
        self.get_logger().info(f"New goal set: {self.goal}")

    # ---------------------------
    # Main control loop (timer callback)
    # ---------------------------
    def control_loop(self):
        # run DWA step at configured Hz
        if self.pose is None or self.current_scan_sectors is None or self.goal is None:
            # nothing to do until we have pose, scan and goal
            return

        self.control_step += 1

        # Timeout
        if self.control_step > self.max_control_steps:
            self.get_logger().warn('Timeout reached -> stopping')
            self.stop_robot()
            return

        # Publish intermediate feedback every feedback_rate steps
        if self.control_step % self.feedback_rate == 0:
            dist = np.linalg.norm(self.goal - self.pose[0:2])
            msg = Float32()
            msg.data = float(dist)
            self.feedback_pub.publish(msg)
            self.get_logger().info(f"[Feedback] distance to goal: {dist:.3f} m (step {self.control_step})")

        # Safety check: immediate stop if any sector reading < collision threshold
        with self.scan_lock:
            min_range = float(np.min(self.current_scan_sectors)) if self.current_scan_sectors is not None else float('inf')
        if min_range < self.collision_threshold:
            self.get_logger().warn(f"Collision imminent (min_range={min_range:.3f} < threshold={self.collision_threshold:.3f}) -> STOP")
            self.stop_robot()
            return

        # Compute DWA control
        v_cmd, w_cmd = self.compute_dwa_cmd()

        # Publish cmd_vel
        twist = Twist()
        twist.linear.x = float(v_cmd)
        twist.angular.z = float(w_cmd)
        self.cmd_pub.publish(twist)

        # Check goal reached
        dist_to_goal = np.linalg.norm(self.goal - self.pose[0:2])
        if dist_to_goal < 0.15:
            self.get_logger().info("Goal reached ✔")
            self.stop_robot()
            # reset goal so loop idles until new goal
            self.goal = None

    # ---------------------------
    # DWA core
    # ---------------------------
    def compute_dwa_cmd(self):
        # 1) Compute dynamic window bounds (based on current velocity and accel limits)
        min_v, max_v, min_w, max_w = self.compute_dynamic_window(self.vel)

        # 2) Sample velocities in window
        v_values = np.linspace(min_v, max_v, self.v_samples)
        w_values = np.linspace(min_w, max_w, self.w_samples)

        best_score = -1e9
        best_cmd = (0.0, 0.0)

        # Precompute obstacle coordinates from current scan sectors (in odom frame)
        obst_points = self.scan_sectors_to_points()

        # For each candidate velocity pair simulate trajectory and evaluate
        for v in v_values:
            for w in w_values:
                traj = self.simulate_trajectory(self.pose, v, w)
                if traj is None:
                    continue
                # if any point collides with obstacle (within robot radius), discard (very negative score)
                min_dist_traj = self.min_dist_to_obstacles(traj, obst_points)
                if min_dist_traj < self.robot_radius + 0.0:  # collision (no tolerance here; immediate discard)
                    continue

                # Evaluate cost components
                heading_score = self.score_heading(traj)
                vel_score = self.score_velocity(v)
                obs_score = self.score_obstacle_distance(min_dist_traj)

                # Normalize each into comparable range before weighting
                # heading_score in [0,1], vel_score [0,1], obs_score [0,1]
                # final objective: maximize J
                J = (self.alpha * heading_score) + (self.beta * vel_score) + (self.gamma * obs_score)

                if J > best_score:
                    best_score = J
                    best_cmd = (v, w)

        # If no candidate found (rare), stop
        if best_score < -1e8:
            return 0.0, 0.0
        return best_cmd

    def compute_dynamic_window(self, current_vel):
        """
        Compute reachable velocities considering acceleration limits:
        Vd = [v_curr - a_v*dt, v_curr + a_v*dt] intersect [min_lin_vel, max_lin_vel]
             similarly for angular velocity.
        """
        v_curr, w_curr = float(current_vel[0]), float(current_vel[1])
        min_v = max(self.min_lin_vel, v_curr - self.max_lin_acc * self.dt)
        max_v = min(self.max_lin_vel, v_curr + self.max_lin_acc * self.dt)
        min_w = max(-self.max_ang_vel, w_curr - self.max_ang_acc * self.dt)
        max_w = min(self.max_ang_vel, w_curr + self.max_ang_acc * self.dt)
        return min_v, max_v, min_w, max_w

    def simulate_trajectory(self, start_pose, v, w):
        """
        Simulate robot trajectory for sim_time with constant (v,w).
        start_pose: np.array([x,y,yaw])
        returns trajectory: np.array shape (n_steps, 3)
        """
        t_steps = int(max(1, round(self.sim_time / self.time_granularity)))
        traj = np.zeros((t_steps, 3))
        x, y, yaw = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
        dt_sim = self.time_granularity
        for i in range(t_steps):
            # simple differential drive kinematic update (constant v,w)
            x += v * math.cos(yaw) * dt_sim
            y += v * math.sin(yaw) * dt_sim
            yaw += w * dt_sim
            yaw = self.normalize_angle(yaw)
            traj[i, :] = [x, y, yaw]
        return traj

    def scan_sectors_to_points(self):
        """
        Convert each sector (distance + angle relative to robot) into an (x,y) in odom frame,
        using current robot pose.
        """
        pts = []
        with self.scan_lock:
            sectors = self.current_scan_sectors
            angles = self.scan_angles
            pose = self.pose.copy() if self.pose is not None else None
        if sectors is None or angles is None or pose is None:
            return np.zeros((0,2))
        rx, ry, ryaw = pose
        for d, ang in zip(sectors, angles):
            # angle is relative to robot heading; transform to odom
            gx = rx + d * math.cos(ryaw + ang)
            gy = ry + d * math.sin(ryaw + ang)
            pts.append([gx, gy])
        return np.array(pts)

    def min_dist_to_obstacles(self, traj, obst_pts):
        """
        Compute minimum Euclidean distance between a simulated trajectory (sequence of x,y)
        and the set of obstacle points.
        """
        if traj is None or traj.shape[0] == 0:
            return 0.0
        if obst_pts is None or obst_pts.shape[0] == 0:
            return 1e3
        px = traj[:, 0:2][:, None, :]  # (T,1,2)
        obs = obst_pts[None, :, :]     # (1,N,2)
        diff = px - obs                # (T,N,2)
        dists = np.linalg.norm(diff, axis=-1)  # (T,N)
        min_dist = float(np.min(dists))
        return min_dist

    # ---------------------------
    # Scoring / normalization helpers
    # ---------------------------
    def score_heading(self, traj):
        # heading: how aligned is final pose to goal direction
        if traj is None or traj.shape[0] == 0 or self.goal is None:
            return 0.0
        last = traj[-1]
        dx = self.goal[0] - last[0]
        dy = self.goal[1] - last[1]
        goal_angle = math.atan2(dy, dx)
        diff = abs(self.normalize_angle(goal_angle - last[2]))
        # convert angle difference to score in [0,1] (1 is perfectly aligned)
        score = (math.pi - diff) / math.pi
        return max(0.0, min(1.0, score))

    def score_velocity(self, v):
        # prefer higher forward velocity; normalize by max_lin_vel
        score = float(v / (self.max_lin_vel + 1e-6))
        return max(0.0, min(1.0, score))

    def score_obstacle_distance(self, dist):
        # Convert distance to [0,1], saturate at obstacle_max (we take lidar_max_range)
        score = float(dist / (self.lidar_max_range + 1e-6))
        return max(0.0, min(1.0, score))

    # ---------------------------
    # Utils
    # ---------------------------
    @staticmethod
    def quaternion_to_yaw(q):
        # q is geometry_msgs/Quaternion
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def normalize_angle(a):
        # normalize to [-pi, pi]
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

def main(args=None):
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
