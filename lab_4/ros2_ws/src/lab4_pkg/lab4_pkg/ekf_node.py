#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from angles import normalize_angle
from math import sin, cos, atan2, sqrt

class EKFNode(Node):

    def __init__(self):
        super().__init__('ekf_node')
        
        # --- State and Covariance ---
        # State mu = [x, y, theta]
        self.mu = np.array([0.0, 0.0, 0.0])
        # Covariance Sigma = P
        self.Sigma = np.diag([0.1, 0.1, 0.1])

        # --- EKF Noise Parameters ---
        # Motion Noise (in control space) [a1, a2, a3, a4]
        # These correspond to alpha_1..alpha_4 in the velocity model
        self.a = [0.1, 0.01, 0.01, 0.1] # TODO: Tune these!
        
        # Measurement Noise (in measurement space)
        # [std_dev_range, std_dev_bearing]
        self.Q = np.diag([
            0.1**2,       # std_dev_range^2 (10cm)
            (1.0*np.pi/180.0)**2  # std_dev_bearing^2 (1 degree in radians)
        ])

        # --- Landmark Map ---
        # Based on Table 1
        self.landmarks = {
            #11: np.array([-1.1, -1.1]),
            #12: np.array([-1.1, 0.0]),
            #13: np.array([-1.1, 1.1]),
           # 21: np.array([0.0, -1.1]),
            #22: np.array([0.0, 0.0]),
            #23: np.array([0.0, 1.1]),
           # 31: np.array([1.1, -1.1]),
           # 32: np.array([1.1, 0.0]),
           # 33: np.array([1.1, 1.1]),
           # --- Landmark Map (REAL ROBOT - IDs 0 to 7) ---
        # Coordinates from '04_05_06_Test on real robot rules.pdf', Page 7
            0: np.array([1.20, 1.68]),
            1: np.array([1.68, -0.05]),
            2: np.array([3.72, 0.14]),
            3: np.array([3.75, 1.37]),
            4: np.array([2.48, 1.25]),
            5: np.array([4.80, 1.87]),
            6: np.array([2.18, 1.00]),
            7: np.array([2.94, 2.70]),
        }

        # --- ROS Components ---
        self.last_odom = None
        self.last_update_time = self.get_clock().now()

        # Publisher for EKF pose
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10) # [cite: 52]

        # Subscriber to odometry (for controls)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10) # [cite: 43]

        # Subscriber to landmark measurements
        self.landmark_sub = self.create_subscription(
            LandmarkArray, '/landmarks', self.update_step, 10) # [cite: 48]

        # Timer for prediction step (20 Hz)
        self.timer = self.create_timer(1.0 / 20.0, self.prediction_step) # [cite: 42, 45]

    def odom_callback(self, msg):
        # Store the latest odometry message
        self.last_odom = msg

    def prediction_step(self):
        # --- 1. Get Control and dt ---
        if self.last_odom is None:
            return

        # Get controls [v, w] from the /odom topic [cite: 43]
        v = self.last_odom.twist.twist.linear.x
        w = self.last_odom.twist.twist.angular.z
        
        # Calculate dt
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = now
        
        if dt < 0.001:
            return

        # --- 2. Calculate Motion Model and Jacobians ---
        x, y, theta = self.mu
        
        # Handle the w=0 case (straight line motion)
        if abs(w) < 1e-3:
            # Non-linear model g(x, u)
            g = self.mu + np.array([
                v * cos(theta) * dt,
                v * sin(theta) * dt,
                0.0
            ])
            # Jacobian G w.r.t state x
            G = np.array([
                [1.0, 0.0, -v * sin(theta) * dt],
                [0.0, 1.0,  v * cos(theta) * dt],
                [0.0, 0.0, 1.0]
            ])
            # Jacobian V w.r.t control u
            V = np.array([
                [cos(theta) * dt, 0.0],
                [sin(theta) * dt, 0.0],
                [0.0, dt]
            ])
        else:
            # Non-linear model g(x, u) (from Task 0)
            g = self.mu + np.array([
                -v/w * sin(theta) + v/w * sin(theta + w * dt),
                 v/w * cos(theta) - v/w * cos(theta + w * dt),
                 w * dt
            ])
            # Jacobian G w.r.t state x (from Task 0)
            G = np.array([
                [1.0, 0.0, -v/w * cos(theta) + v/w * cos(theta + w * dt)],
                [0.0, 1.0,  v/w * sin(theta) - v/w * sin(theta + w * dt)],
                [0.0, 0.0, 1.0]
            ])
            # Jacobian V w.r.t control u (from Task 0)
            V = np.array([
                [(-sin(theta) + sin(theta + w * dt)) / w,
                 v * (sin(theta) - sin(theta + w * dt)) / w**2 + v * dt * cos(theta + w * dt) / w],
                [(cos(theta) - cos(theta + w * dt)) / w,
                 -v * (cos(theta) - cos(theta + w * dt)) / w**2 + v * dt * sin(theta + w * dt) / w],
                [0.0, dt]
            ])

        # --- 3. Calculate Noise Covariance M ---
        # Covariance in control space
        M = np.array([
            [self.a[0]*v**2 + self.a[1]*w**2, 0.0],
            [0.0, self.a[2]*v**2 + self.a[3]*w**2]
        ])
        
        # --- 4. Predict State and Covariance ---
        # Predicted state mu_bar
        self.mu = g
        self.mu[2] = normalize_angle(self.mu[2]) # Normalize angle
        
        # Predicted covariance Sigma_bar
        self.Sigma = G @ self.Sigma @ G.T + V @ M @ V.T

    def update_step(self, msg: LandmarkArray):
        # [cite: 51]
        
        mu_bar = self.mu
        Sigma_bar = self.Sigma
        
        # Loop through each detected landmark in the message
        for landmark_msg in msg.landmarks:
            landmark_id = landmark_msg.id
            
            # Check if we know this landmark
            if landmark_id not in self.landmarks:
                continue

            # Get landmark's true position
            m = self.landmarks[landmark_id]
            mx, my = m[0], m[1]
            
            # Get measurement z = [range, bearing]
            z = np.array([landmark_msg.range, landmark_msg.bearing])

            # --- 1. Calculate Measurement Model and Jacobian H ---
            x, y, theta = mu_bar
            q_sq = (mx - x)**2 + (my - y)**2
            q = sqrt(q_sq)

            # Predicted measurement h(mu_bar)
            h = np.array([
                q,
                normalize_angle(atan2(my - y, mx - x) - theta)
            ])
            
            # Jacobian H w.r.t state (from Task 0)
            H = np.array([
                [-(mx - x) / q, -(my - y) / q, 0.0],
                [ (my - y) / q_sq, -(mx - x) / q_sq, -1.0]
            ])

            # --- 2. Compute Kalman Gain K ---
            S = H @ Sigma_bar @ H.T + self.Q
            K = Sigma_bar @ H.T @ np.linalg.inv(S)

            # --- 3. Correct State and Covariance ---
            # Innovation (measurement error)
            y = z - h
            y[1] = normalize_angle(y[1]) # CRITICAL: normalize bearing error
            
            # Corrected state mu
            mu_bar = mu_bar + K @ y
            
            # Corrected covariance Sigma
            Sigma_bar = (np.eye(3) - K @ H) @ Sigma_bar
        
        # --- 4. Save Corrected State ---
        self.mu = mu_bar
        self.mu[2] = normalize_angle(self.mu[2])
        self.Sigma = Sigma_bar
        
        # --- 5. Publish the EKF result ---
        self.publish_odometry(msg.header.stamp) # [cite: 52]

    def publish_odometry(self, stamp):
        """Publishes the EKF state as a nav_msgs/msg/Odometry message."""
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp # [cite: 53]
        odom_msg.header.frame_id = "odom"       # Global frame
        odom_msg.child_frame_id = "base_link"   # Robot frame

        # Set position
        odom_msg.pose.pose.position.x = self.mu[0]
        odom_msg.pose.pose.position.y = self.mu[1]
        
        # Set orientation (as quaternion)
        q = quaternion_from_euler(0.0, 0.0, self.mu[2])
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        # Set covariance
        # This is a 6x6 matrix, we only fill the 3x3 part for x, y, yaw
        pose_cov = np.zeros((6, 6))
        pose_cov[0:2, 0:2] = self.Sigma[0:2, 0:2] # x, y
        pose_cov[5, 5] = self.Sigma[2, 2]         # yaw
        pose_cov[0, 5] = self.Sigma[0, 2]         # x-yaw
        pose_cov[5, 0] = self.Sigma[2, 0]         # yaw-x
        pose_cov[1, 5] = self.Sigma[1, 2]         # y-yaw
        pose_cov[5, 1] = self.Sigma[2, 1]         # yaw-y
        odom_msg.pose.covariance = pose_cov.flatten().tolist()
        
        # Pass through velocity from odom for visualization
        if self.last_odom is not None:
            odom_msg.twist.twist = self.last_odom.twist.twist

        self.ekf_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    ekf_node = EKFNode()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()