import numpy as np
import math
import yaml
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from landmark_msgs.msg import LandmarkArray
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory

# Local imports
from .task1 import EKFNode, load_landmarks # Re-use load_landmarks from task1 if possible, or define here
from .EKF import RobotEKF
from .velocity_model4task2 import motion_model_wrapper, normalize_angle

# Re-definition if you want task2 to be standalone, otherwise import is fine.
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
        # We need to initialize the Node first, but we are overriding __init__
        # to set up specific paths and 5-state logic.
        
        # NOTE: calling super().__init__() would run Task1 logic. 
        # Instead, we behave like a Node directly but inherit helper methods.
        Node.__init__(self, 'extended_ekf_node') 

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
        self.ekf.eval_gux = motion_model_wrapper
        self.ekf.eval_Gt = self.jacobian_Gt_extended
        self.ekf.eval_Vt = self.jacobian_Vt_extended

        # Initial state and covariance
        self.ekf.mu = np.zeros(5)  # [x, y, theta, v, omega]
        self.ekf.Sigma = np.diag([0.50**2, 0.50**2, math.radians(10)**2, 0.25**2, 0.25**2])
        self.ekf._I = np.eye(5)
        
        # Tuning parameters
        self.alpha = np.array([0.001, 0.01, 0.1, 0.2, 0.05, 0.05])
        self.sigma_u = np.array([0.50, 0.30]) # Used for process noise

        # Measurement noise tuning
        self.Qt_landmark = np.diag([0.02**2, (math.pi / 36)**2])
        self.Qt_odom     = np.diag([0.05**2, 0.05**2])
        self.Qt_imu      = np.diag([0.08**2])

        # --- RUNTIME STATE ---
        self.initialized = False
        self.first_odom = None
        self.last_cmd = np.array([0.0, 0.0])
        self.last_time = self.get_clock().now()
        
        if not self.initialized:
            self.ekf.mu = np.array([-2.0, -0.5, 0.0, 0.0, 0.0])
            self.initialized = True

        # --- SUBSCRIPTIONS ---
        self.create_subscription(Odometry,       self.odom_topic,      self.odom_callback,      10)
        self.create_subscription(Imu,            self.imu_topic,       self.imu_callback,       10)
        self.create_subscription(LandmarkArray,  self.landmarks_topic, self.landmarks_callback, 10)
        
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("ExtendedEKFNode initialized.")

    # =======================
    # Motion Jacobians (5x5)
    # =======================
    def jacobian_Gt_extended(self, mu, u_unused, dt):
        x, y, th, v, w = mu
        
        # --- ROBUST HANDLING FOR w=0 ---
        # If w is very small, use the derivative of the straight-line model
        # (L'Hopital's rule limit) to avoid dividing by zero.
        if abs(w) < 1e-5:
            # Derivatives of x' = x + v*cos(th)*dt
            # Derivatives of y' = y + v*sin(th)*dt
            sin_th = math.sin(th)
            cos_th = math.cos(th)
            
            return np.array([
                [1, 0, -v * sin_th * dt,  cos_th * dt, 0],
                [0, 1,  v * cos_th * dt,  sin_th * dt, 0],
                [0, 0, 1,                 0,           dt],
                [0, 0, 0,                 1,           0],
                [0, 0, 0,                 0,           1]
            ])
            
        else:
            # Use exact Arc motion model derivatives
            dtheta = w * dt
            sin_th = math.sin(th)
            cos_th = math.cos(th)
            sin_th_dt = math.sin(th + dtheta)
            cos_th_dt = math.cos(th + dtheta)
            
            # Common terms
            r = v / w
            
            # 1. Derivative w.r.t theta (d/dth)
            # dx/dth = -r*cos(th) + r*cos(th+dt*w)
            dp_dth_x = -r * cos_th + r * cos_th_dt
            dp_dth_y = -r * sin_th + r * sin_th_dt
            
            # 2. Derivative w.r.t v (d/dv)
            # dx/dv = 1/w * (-sin(th) + sin(th+dt*w))
            term_x = -sin_th + sin_th_dt
            term_y =  cos_th - cos_th_dt
            dp_dv_x = term_x / w
            dp_dv_y = term_y / w

            # 3. Derivative w.r.t w (d/dw)
            # Product rule on v/w * term
            # d/dw = (-v/w^2) * term + (v/w) * d(term)/dw
            # d(term_x)/dw = cos(th+dt*w) * dt
            dp_dw_x = -(v / w**2) * term_x + (v / w) * (cos_th_dt * dt)
            dp_dw_y = -(v / w**2) * term_y + (v / w) * (sin_th_dt * dt)
            
            return np.array([
                [1, 0, dp_dth_x, dp_dv_x, dp_dw_x],
                [0, 1, dp_dth_y, dp_dv_y, dp_dw_y],
                [0, 0, 1,        0,       dt     ],
                [0, 0, 0,        1,       0      ],
                [0, 0, 0,        0,       1      ]
            ])

    def jacobian_Vt_extended(self, mu, u_unused, dt):
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
        v_cmd = msg.twist.twist.linear.x
        w_cmd = msg.twist.twist.angular.z
        self.last_cmd = np.array([v_cmd, w_cmd])

        if not self.initialized and self.first_odom is None:
            # Initialization logic same as before...
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "odom"
            odom_msg.pose.pose.position = Point(x=0.0, y=0.77, z=0.0)
            theta = 0.0
            odom_msg.pose.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=math.sin(theta/2.0), w=math.cos(theta/2.0)
            )
            self.first_odom = odom_msg
            self.ekf.mu = np.array([
                self.first_odom.pose.pose.position.x, 
                self.first_odom.pose.pose.position.y, 
                theta, 0, 0
            ])
            self.initialized = True

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
        if not self.initialized:
            # Logic to init from landmark if odom is not sufficient...
            pass # (kept brief, logic is same as original file)

        if self.initialized and msg.landmarks:
            for lm in msg.landmarks:
                lm_id = lm.id
                if lm_id not in self.landmark_ids:
                    continue
                idx = self.landmark_ids.index(lm_id)
                landmark_pos = self.landmarks[idx]
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

            v_var = alpha[0]*v_cmd**2 + alpha[1]*w_cmd**2
            w_var = alpha[2]*v_cmd**2 + alpha[3]*w_cmd**2
            
            self.ekf.Mt = np.diag([v_var, w_var])

            self.ekf.predict(
                u=self.last_cmd,
                sigma_u=self.sigma_u,
                g_extra_args=(dt,)
            )

            self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])
            self.get_logger().debug(f"EKF Prediction: dt={dt:.3f}s, u={self.last_cmd}")

        self.publish_estimated_state()

    # =======================
    # Measurement models (Task 2)
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
        sq = math.sqrt(max(q, 1e-9))
        return np.array([
            [-dx / sq, -dy / sq, 0, 0, 0],
            [ dy / max(q, 1e-9), -dx / max(q, 1e-9), -1, 0, 0]
        ])

    def h_odom(self, mu):
        return mu[3:5]

    def Ht_odom(self, mu):
        return np.array([
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])

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