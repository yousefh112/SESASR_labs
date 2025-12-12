import numpy as np
import math
import yaml
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from landmark_msgs.msg import LandmarkArray
from .task1 import EKFNode
from .EKF import RobotEKF
from .velocity_model4task2 import motion_model_wrapper, normalize_angle
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory


def load_landmarks(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    ids = data['landmarks']['id']
    xs = data['landmarks']['x']
    ys = data['landmarks']['y']
    positions = [[x, y] for x, y in zip(xs, ys)]
    return ids, positions


class ExtendedEKFNode(EKFNode):
    """EKF with extended state: [x, y, θ, v, ω] (Option A: velocities are estimated)."""

    def __init__(self):
        super().__init__()

       
        self.flip_landmark_bearing = False 
        self.landmarks_topic = "/landmarks"
        self.odom_topic = "/odom"
        self.imu_topic = "/imu"
        
        # --- PATH CONFIGURATION ---
        package_share_directory = get_package_share_directory('lab04_pkg')
        yaml_file = os.path.join(package_share_directory, 'config', 'landmarks.yaml')
        
        self.get_logger().info("Extended EKF (Option A): state = [x, y, θ, v, ω]")
        self.get_logger().info(f"Loading landmarks from: {yaml_file}")
        
        # --- LANDMARKS ---
        self.landmark_ids, self.landmarks = load_landmarks(yaml_file)
        self.get_logger().info(f"Loaded {len(self.landmarks)} landmarks from YAML.")

        # --- EKF OBJECT: ensure 5-state configuration ---
        self.ekf = RobotEKF(initial_mu=[-2,-0.5,0])
        self.ekf.dim_x = 5
        self.ekf.dim_u = 2
        self.ekf.eval_gux = motion_model_wrapper    # uses state v,ω for kinematics; does not overwrite them
        self.ekf.eval_Gt = self.jacobian_Gt_extended
        self.ekf.eval_Vt = self.jacobian_Vt_extended

        # Initial state and covariance (moderately uncertain)
        self.ekf.mu = np.zeros(5)  # [x, y, theta, v, omega]
        self.ekf.Sigma = np.diag([0.50**2, 0.50**2, math.radians(10)**2, 0.25**2, 0.25**2])
        self.ekf._I = np.eye(5)

        # Process/input noise for controls (std devs for v_cmd, w_cmd) injected via Vt into state
        # In Option A, controls are not used to set v, ω; they only define process noise magnitude
        self.sigma_u = np.array([0.50, 0.30])

        # Measurement noise tuning (std devs)
        self.Qt_landmark = np.diag([0.02**2, (math.pi / 36)**2])   # range meters, bearing ~5°
        self.Qt_odom     = np.diag([0.05**2, 0.05**2])             # v, ω from wheel odom
        self.Qt_imu      = np.diag([0.08**2])                      # ω from IMU

        # --- RUNTIME STATE ---
        self.initialized = False
        self.first_odom = None
        self.last_cmd = np.array([0.0, 0.0])  # [v_cmd, w_cmd] used only for process noise scaling
        self.last_time = self.get_clock().now()
        if not self.initialized:
            self.ekf.mu = np.array([-2.0, -0.5, 0.0,0.0,0.0])
            self.initialized = True

        # --- SUBSCRIPTIONS ---
        self.create_subscription(Odometry,       self.odom_topic,      self.odom_callback,      10)
        self.create_subscription(Imu,            self.imu_topic,       self.imu_callback,       10)
        self.create_subscription(LandmarkArray,  self.landmarks_topic, self.landmarks_callback, 10)

        self.get_logger().info("ExtendedEKFNode initialized.")

    # =======================
    # Motion Jacobians (5x5)
    # =======================
    def jacobian_Gt_extended(self, mu, u_unused, dt):
        # Extract state
        x, y, th, v, w = mu

        # Compute change in orientation
        dtheta = w * dt

        # Apply the Jacobian from the professor's book (image)
        return np.array([
            [1, 0, -v * math.cos(th) / w + v * math.cos(th + dtheta / 2) / w, 0, 0],
            [0, 1, -v * math.sin(th) / w + v * math.sin(th + dtheta / 2) / w, 0, 0],
            [0, 0, 1,                                                   0, dt],
            [0, 0, 0,                                                   1, 0],
            [0, 0, 0,                                                   0, 1]
        ])


    def jacobian_Vt_extended(self, mu, u_unused, dt):
        # Maps control/process noise (on v_cmd, w_cmd) into the state
        x, y, th, v, w = mu
        return np.array([
            [math.cos(th) * dt, 0],
            [math.sin(th) * dt, 0],
            [0,                 dt],
            [1,                 0],
            [0,                 1]
        ])

    # =======================
    # Callbacks
    # =======================
    def odom_callback(self, msg: Odometry):
        # Store commands to scale process noise (Option A: commands do not set v, ω directly)
        v_cmd = msg.twist.twist.linear.x
        w_cmd = msg.twist.twist.angular.z
        self.last_cmd = np.array([v_cmd, w_cmd])

        # Save the first odom message for orientation on init
        if not self.initialized and self.first_odom is None:
            odom_msg = Odometry()

            # Fill in the header (optional but recommended)
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "odom"

            # Set the pose
            odom_msg.pose.pose.position = Point(x=0.0, y=0.77, z=0.0)

            # If you know the orientation as a yaw angle (theta)
            theta = 0.0  # radians
            odom_msg.pose.pose.orientation = Quaternion(
                x=0.0,
                y=0.0,
                z=math.sin(theta / 2.0),
                w=math.cos(theta / 2.0)
)

            # Save it as your first odom
            self.first_odom = odom_msg
            self.ekf.mu = np.array([self.first_odom.pose.pose.position.x, self.first_odom.pose.pose.position.y, theta, 0,0])
            self.initialized = True

        # Update the v, ω substate from odometry measurement (if initialized)
        if self.initialized:
            z = np.array([v_cmd, w_cmd])
            self.ekf.update(
                z=z,
                eval_hx=self.h_odom,
                eval_Ht=self.Ht_odom,
                Qt=self.Qt_odom,
                Ht_args=(self.ekf.mu,),
                hx_args=(self.ekf.mu,)
            )

    def imu_callback(self, msg: Imu):
        # IMU angular velocity → updates omega only
        omega_meas = msg.angular_velocity.z
        z = np.array([omega_meas])

        if self.initialized:
            self.ekf.update(
                z=z,
                eval_hx=self.h_imu,
                eval_Ht=self.Ht_imu,
                Qt=self.Qt_imu,
                Ht_args=(self.ekf.mu,),
                hx_args=(self.ekf.mu,)
            )

    def landmarks_callback(self, msg: LandmarkArray):
        # Initialization using the first available landmark, after first odom is received
        if not self.initialized:
            if self.first_odom is None:
                self.get_logger().warn("Skipping landmark: waiting for first odometry message.")
                return
            if not msg.landmarks:
                return

            lm = msg.landmarks[0]
            lm_id = lm.id

            # Lookup global landmark pose
            if lm_id not in self.landmark_ids:
                self.get_logger().warn(f"Unknown landmark ID {lm_id}")
                return

            idx = self.landmark_ids.index(lm_id)
            landmark_pos = np.array(self.landmarks[idx])  # [mx, my]

            # Heading from first odom quaternion (assuming near-zero roll/pitch)
            q = self.first_odom.pose.pose.orientation
            odom_theta = 2.0 * math.atan2(q.z, q.w)
            odom_theta = normalize_angle(odom_theta)

            # Measurement range and bearing
            r = lm.range
            b = lm.bearing
            b_eff = -b if self.flip_landmark_bearing else b

            # Robot global pose = landmark_global - measurement_in_global_frame
            robot_x = landmark_pos[0] - r * math.cos(odom_theta + b_eff)
            robot_y = landmark_pos[1] - r * math.sin(odom_theta + b_eff)

            # Initialize velocities to last commands (reasonable starting guess)
            v0 = self.last_cmd[0]
            w0 = self.last_cmd[1]

            self.ekf.mu = np.array([robot_x, robot_y, odom_theta, v0, w0])
            self.ekf.Sigma = np.diag([0.50**2, 0.50**2, math.radians(10)**2, 0.25**2, 0.25**2])
            self.initialized = True
            self.get_logger().info(f"EKF initialized in map frame: {self.ekf.mu}")

        # Regular landmark updates after initialization
        if self.initialized and msg.landmarks:
            for lm in msg.landmarks:
                lm_id = lm.id
                if lm_id not in self.landmark_ids:
                    continue

                idx = self.landmark_ids.index(lm_id)
                landmark_pos = self.landmarks[idx]  # [mx, my]
                r = lm.range
                b = lm.bearing
                b_eff = -b if self.flip_landmark_bearing else b

                z = np.array([r, b_eff])

                self.ekf.update(
                    z=z,
                    eval_hx=self.h_landmark,
                    eval_Ht=self.Ht_landmark,
                    Qt=self.Qt_landmark,
                    Ht_args=(self.ekf.mu, landmark_pos),
                    hx_args=(self.ekf.mu, landmark_pos)
                )
    def timer_callback(self):
            now = self.get_clock().now()
            dt = (now - self.last_time).nanoseconds / 1e9
            self.last_time = now

            if self.initialized:
                v_cmd = self.last_cmd[0]
                w_cmd = self.last_cmd[1]
                alpha = self.alpha

                # --- CRITICAL FIX: Dynamic calculation of Mt ---
                # Mt = diag([v_variance, w_variance]), using alpha1 to alpha4
                v_var = alpha[0]*v_cmd**2 + alpha[1]*w_cmd**2
                w_var = alpha[2]*v_cmd**2 + alpha[3]*w_cmd**2
                
                # The angular drift terms (alpha5, alpha6) are absorbed into the 
                # uncertainty propagation if Vt and Gt are fully defined.
                self.ekf.Mt = np.diag([v_var, w_var])
                # -----------------------------------------------

                # Predict using current state velocities
                # u=self.last_cmd is passed to allow the EKF.predict to access command velocities
                # for the Mt calculation inside the EKF if necessary (though we do it here).
                self.ekf.predict(
                    u=self.last_cmd,sigma_u=self.sigma_u,
                    # Removed sigma_u argument as it's redundant/misleading when using Mt
                    g_extra_args=(dt,)
                )

                # Keep theta bounded
                self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])

                self.get_logger().debug(f"EKF Prediction: dt={dt:.3f}s, u={self.last_cmd}")

            self.publish_estimated_state()



    # =======================
    # Measurement models
    # =======================
    def h_landmark(self, mu, landmark):
        x, y, th, v, w = mu
        mx, my = landmark
        dx = mx - x
        dy = my - y
        r = math.sqrt(dx * dx + dy * dy)
        phi_raw = math.atan2(dy, dx) - th
        phi = normalize_angle(phi_raw)
        return np.array([r, phi])

    def Ht_landmark(self, mu, landmark):
        x, y, th, v, w = mu
        mx, my = landmark
        dx = mx - x
        dy = my - y
        q = dx * dx + dy * dy
        sq = math.sqrt(max(q, 1e-9))  # guard against division by zero
        return np.array([
            [-dx / sq, -dy / sq, 0, 0, 0],
            [ dy / max(q, 1e-9), -dx / max(q, 1e-9), -1, 0, 0]
        ])

    # Odom updates v, ω only
    def h_odom(self, mu):
        return mu[3:5]

    def Ht_odom(self, mu):
        return np.array([
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])

    # IMU updates ω only
    def h_imu(self, mu):
        return np.array([mu[4]])

    def Ht_imu(self, mu):
        return np.array([[0, 0, 0, 0, 1]])


def main(args=None):
    rclpy.init(args=args)
    node = ExtendedEKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
