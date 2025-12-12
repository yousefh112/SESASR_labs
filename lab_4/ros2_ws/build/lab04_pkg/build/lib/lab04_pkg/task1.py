import rclpy 
from rclpy.node import Node 
from nav_msgs.msg import Odometry 
from landmark_msgs.msg import LandmarkArray
import numpy as np 
import math 
import yaml 
import os
from ament_index_python.packages import get_package_share_directory
# Assuming imports below are correct based on your package structure
from .EKF import RobotEKF 
from .velocity4ekf import velocity_motion_model_wrapper, jacobian_Gt, jacobian_Vt
from .landmark_model import landmark_model_jacobian


def load_landmarks(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    ids = data['landmarks']['id']
    xs = data['landmarks']['x']
    ys = data['landmarks']['y']

    positions = [[x, y] for x, y in zip(xs, ys)]

    return ids, positions


class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        # --- PATH CONFIGURATION ---
        # Dynamically find the path to the config folder in the installed package
        package_share_directory = get_package_share_directory('lab04_pkg')
        yaml_file = os.path.join(package_share_directory, 'config', 'landmarks.yaml')
        
        self.get_logger().info(f"Loading landmarks from: {yaml_file}")
        self.landmark_ids, self.landmarks = load_landmarks(yaml_file)
        self.get_logger().info(f"Loaded {len(self.landmarks)} landmarks from YAML.")

        # ---- EKF setup ----
        self.ekf = RobotEKF(initial_mu=[-2.0, -0.5, 0.0]) # when real robot [ 0 0.77 0]

        # Tuning Parameters
        self.alpha = np.array([0.001, 0.01, 0.1, 0.2, 0.05, 0.05])
        self.ekf.Mt = np.eye(2) 
        self.sigma_u = np.array([0.1, 0.1])
        self.sigma_z = np.array([0.3, math.pi / 18])
        self.Qt = np.diag(self.sigma_z**2)

        # Initial EKF uncertainty
        self.ekf.Sigma = np.diag([1.0, 1.0, math.radians(30)**2])




        self.last_cmd = np.array([0.0, 0.0])
        self.last_time = self.get_clock().now()
        self.initialized = False
        if not self.initialized:
            self.ekf.mu = np.array([-2.0, -0.5, 0.0])
            self.initialized = True
      
        
        # --- END FORCED INITIALIZATION ---


        # ---- ROS interfaces ----
        # Odom callback is only used to capture v/omega commands now
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10) 
        # Landmark callback is only used for updates now
        self.create_subscription(LandmarkArray, '/landmarks', self.landmarks_callback, 10) 
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        self.create_timer(0.05, self.timer_callback)  # 20 Hz prediction

        self.get_logger().info('EKF node started with prediction + update.')
        
    # ----------------------------
    # Prediction callbacks
    # ----------------------------
    def odom_callback(self, msg):
        # Only capture commands, no initialization logic needed here anymore
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        self.last_cmd = np.array([v, omega])
            
    def timer_callback(self):
        # Initialization is guaranteed, so we proceed directly
        now = self.get_clock().now() 
        dt = (now - self.last_time).nanoseconds / 1e9 
        self.last_time = now 
        
        v = self.last_cmd[0]
        w = self.last_cmd[1]
        alpha = self.alpha
        if self.initialized:
        
            # --- Dynamic calculation of Mt ---
            v_var = alpha[0]*v**2 + alpha[1]*w**2
            w_var = alpha[2]*v**2 + alpha[3]*w**2
            
            self.ekf.Mt = np.diag([v_var, w_var])
            # ---------------------------------

            # Predict new state and covariance
            self.ekf.predict(
                u=self.last_cmd,sigma_u=self.sigma_u,
                g_extra_args=(dt,),

            )
            
            self.get_logger().debug(f"EKF Prediction: dt={dt:.3f}s, u={self.last_cmd}, Mt_diag={self.ekf.Mt.diagonal()}")
        
            self.publish_estimated_state()

    # ----------------------------
    # Update callbacks
    # ----------------------------
    def landmarks_callback(self, msg: LandmarkArray):
        # Since self.initialized is True in __init__, this runs the update loop immediately
        
        # Process EKF updates
        for lm in msg.landmarks:

            lm_id = lm.id
            if lm_id not in self.landmark_ids:
                self.get_logger().warn(f"Unknown landmark ID {lm_id}")
                continue
            
            idx = self.landmark_ids.index(lm_id)
            landmark_pos = np.array(self.landmarks[idx])

            z = np.array([lm.range, lm.bearing])

            self.ekf.update(
                z=z,
                eval_hx=self.eval_hx,
                eval_Ht=landmark_model_jacobian,
                Qt=self.Qt,
                Ht_args=(self.ekf.mu, landmark_pos),
                hx_args=(self.ekf.mu, landmark_pos),
            )
            
            self.get_logger().debug(f"EKF Update completed using landmark {lm_id}.")

        
# ----------------------------
# Measurement model
# ----------------------------
    def eval_hx(self, mu, landmark):
        """Expected measurement ẑ = [range, bearing] from current state."""
        x, y, theta = mu
        m_x, m_y = landmark
        r_hat = math.sqrt((m_x - x) ** 2 + (m_y - y) ** 2)
        phi_hat = math.atan2(m_y - y, m_x - x) - theta
        # Normalize angle
        phi_hat = math.atan2(math.sin(phi_hat), math.cos(phi_hat))
        return np.array([r_hat, phi_hat])


    
# ----------------------------
# Publishing
# ----------------------------
    def publish_estimated_state(self):
        # Since initialization is guaranteed in __init__, this will always publish
        
        mu = self.ekf.mu
        cov = self.ekf.Sigma

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"

        # Position is set directly from the EKF state
        msg.pose.pose.position.x = float(mu[0])
        msg.pose.pose.position.y = float(mu[1])
        msg.pose.pose.position.z = 0.0

        qz = math.sin(mu[2] / 2.0)
        qw = math.cos(mu[2] / 2.0)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Populate covariance matrix 
        msg.pose.covariance[0] = cov[0, 0]
        msg.pose.covariance[7] = cov[1, 1]
        msg.pose.covariance[35] = cov[2, 2]

        self.ekf_pub.publish(msg)
        self.get_logger().info(
            f"EKF Estimate → x={mu[0]:.2f}, y={mu[1]:.2f}, θ={mu[2]:.2f}"
        )

# ----------------------------
# Main
# ----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


