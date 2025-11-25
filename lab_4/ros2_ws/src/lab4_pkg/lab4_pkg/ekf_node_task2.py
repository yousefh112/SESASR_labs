#!/usr/bin/env python3

import rclpy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from landmark_msgs.msg import LandmarkArray
from tf_transformations import quaternion_from_euler
from angles import normalize_angle
from math import sin, cos, atan2, sqrt, pi

# Import the parent class
from lab4_pkg.ekf_node import EKFNode

class EKFNodeTask2(EKFNode):

    def __init__(self):
        # 1. Initialize the parent class
        # This sets up the publishers, the basic landmark subscriber, and the timer
        super().__init__()
        self.get_logger().info("Switching to Task 2 Extended EKF (5D State)...")

        # --- 2. RECONFIGURE STATE (Resize to 5D) ---
        # State mu = [x, y, theta, v, w]
        self.mu = np.zeros(5)
        # Covariance Sigma (5x5)
        self.Sigma = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])

        # --- 3. DEFINE NOISE PARAMETERS ---
        # Process Noise R (Uncertainty in the constant velocity assumption)
        dt_approx = 1.0/20.0
        self.std_dev_process_v = 0.1  # m/s^2
        self.std_dev_process_w = 0.5  # rad/s^2
        
        # Measurement Noise Q (for new sensors)
        # Note: self.Q (landmarks) is already defined in parent, but we can keep it.
        
        # Odom Update Noise
        std_odom_v = 0.05 # m/s
        std_odom_w = 0.1  # rad/s
        self.Q_odom = np.diag([std_odom_v**2, std_odom_w**2])
        
        # IMU Update Noise
        std_imu_w = 0.05 # rad/s
        self.Q_imu = np.diag([std_imu_w**2])

        # --- 4. RECONFIGURE SUBSCRIBERS ---
        
        # In Task 1, /odom was used for CONTROL (prediction). 
        # In Task 2, /odom is a MEASUREMENT (update).
        # We must destroy the parent's subscription and create a new one.
        if self.odom_sub:
            self.destroy_subscription(self.odom_sub)
            
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_update_callback, 10)
            
        # Add IMU Subscriber
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_update_callback, 10)

    # ---------------------------------------------------------
    # PREDICTION STEP (Overridden)
    # ---------------------------------------------------------
    def prediction_step(self):
        """
        Task 2 Prediction: Constant Velocity Model.
        We do NOT use self.last_odom for control anymore.
        We use the v, w from our own state vector.
        """
        # Calculate dt
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = now
        
        if dt < 0.001 or dt > 1.0:
            return

        # Extract state
        x, y, theta, v, w = self.mu

        # --- Motion Model (Constant Velocity) ---
        # Handle w ~ 0
        if abs(w) < 1e-3:
            # Linear approximation
            g = self.mu + np.array([
                v * cos(theta) * dt,
                v * sin(theta) * dt,
                0.0,
                0.0,
                0.0
            ])
            
            # Jacobian G (5x5)
            G = np.eye(5)
            G[0, 2] = -v * sin(theta) * dt
            G[0, 3] = cos(theta) * dt
            G[1, 2] = v * cos(theta) * dt
            G[1, 3] = sin(theta) * dt
            # Other terms are identity or zero
        else:
            # Full model
            g = self.mu + np.array([
                -v/w * sin(theta) + v/w * sin(theta + w * dt),
                 v/w * cos(theta) - v/w * cos(theta + w * dt),
                 w * dt,
                 0.0,
                 0.0
            ])
            
            # Jacobian G (5x5)
            G = np.eye(5)
            # derivatives w.r.t theta
            G[0, 2] = -v/w * cos(theta) + v/w * cos(theta + w * dt)
            G[1, 2] =  v/w * sin(theta) - v/w * sin(theta + w * dt)
            # derivatives w.r.t v
            G[0, 3] = (-sin(theta) + sin(theta + w * dt)) / w
            G[1, 3] = (cos(theta) - cos(theta + w * dt)) / w
            # derivatives w.r.t w
            G[0, 4] = v * (sin(theta) - sin(theta + w * dt)) / w**2 + v * dt * cos(theta + w * dt) / w
            G[1, 4] = -v * (cos(theta) - cos(theta + w * dt)) / w**2 + v * dt * sin(theta + w * dt) / w
            G[2, 4] = dt

        # --- Process Noise R ---
        R = np.zeros((5, 5))
        R[3, 3] = (self.std_dev_process_v * dt)**2
        R[4, 4] = (self.std_dev_process_w * dt)**2

        # --- Predict ---
        self.mu = g
        self.mu[2] = normalize_angle(self.mu[2])
        self.Sigma = G @ self.Sigma @ G.T + R
        
        # Publish the continuous prediction
        self.publish_odometry(now.to_msg())

    # ---------------------------------------------------------
    # UPDATE STEPS
    # ---------------------------------------------------------
    
    def update_step(self, msg: LandmarkArray):
        """
        Overriding the parent's landmark update to handle 5D state.
        The parent class subscribes '/landmarks' to this method name.
        """
        mu_bar = self.mu
        Sigma_bar = self.Sigma

        for landmark_msg in msg.landmarks:
            lid = landmark_msg.id
            if lid not in self.landmarks:
                continue

            m = self.landmarks[lid]
            mx, my = m[0], m[1]
            z = np.array([landmark_msg.range, landmark_msg.bearing])

            # H Calculation
            x, y, theta = mu_bar[0], mu_bar[1], mu_bar[2]
            q_sq = (mx - x)**2 + (my - y)**2
            q = sqrt(q_sq)

            h = np.array([
                q,
                normalize_angle(atan2(my - y, mx - x) - theta)
            ])

            # Jacobian H (2x5)
            # It's the same as 2x3, but padded with 2 zeros columns
            H = np.zeros((2, 5))
            H[0, 0] = -(mx - x) / q
            H[0, 1] = -(my - y) / q
            H[1, 0] = (my - y) / q_sq
            H[1, 1] = -(mx - x) / q_sq
            H[1, 2] = -1.0
            # Columns 3, 4 (v, w) are zero

            # Correction
            mu_bar, Sigma_bar = self.ekf_correct_general(z, h, H, self.Q, mu_bar, Sigma_bar, is_angle=True)

        self.mu = mu_bar
        self.Sigma = Sigma_bar
        # Parent class publishes here, but we moved publishing to prediction_step 
        # for smoother output, so we can leave it or comment it out.
        # self.publish_odometry(msg.header.stamp)

    def odom_update_callback(self, msg: Odometry):
        """New update step for Task 2 using Odom velocity."""
        z = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        
        # Jacobian H (2x5): Maps state [x,y,th,v,w] to meas [v,w]
        H = np.zeros((2, 5))
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        
        h = H @ self.mu
        
        self.mu, self.Sigma = self.ekf_correct_general(z, h, H, self.Q_odom, self.mu, self.Sigma)

    def imu_update_callback(self, msg: Imu):
        """New update step for Task 2 using IMU angular velocity."""
        z = np.array([msg.angular_velocity.z])
        
        # Jacobian H (1x5): Maps state [x,y,th,v,w] to meas [w]
        H = np.zeros((1, 5))
        H[0, 4] = 1.0
        
        h = H @ self.mu
        
        self.mu, self.Sigma = self.ekf_correct_general(z, h, H, self.Q_imu, self.mu, self.Sigma)

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    def ekf_correct_general(self, z, h, H, Q, mu, Sigma, is_angle=False):
        """Helper to perform the Kalman Correction."""
        S = H @ Sigma @ H.T + Q
        K = Sigma @ H.T @ np.linalg.inv(S)

        y = z - h
        
        # Normalize angle innovation if specifically the second element is an angle
        # (Cheap hack: assumes if is_angle=True, it's the landmark 2D measurement)
        if is_angle and y.size == 2:
            y[1] = normalize_angle(y[1])

        mu_new = mu + K @ y
        Sigma_new = (np.eye(len(mu)) - K @ H) @ Sigma
        
        mu_new[2] = normalize_angle(mu_new[2])
        
        return mu_new, Sigma_new

    def publish_odometry(self, stamp):
        """Overriding to publish 5D state (including Twist)."""
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Pose
        odom_msg.pose.pose.position.x = self.mu[0]
        odom_msg.pose.pose.position.y = self.mu[1]
        q = quaternion_from_euler(0.0, 0.0, self.mu[2])
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        # Twist (The new part of Task 2!)
        odom_msg.twist.twist.linear.x = self.mu[3]
        odom_msg.twist.twist.angular.z = self.mu[4]

        # Covariance (simplified mapping)
        # We map the 5x5 Sigma into the 6x6 ROS msg
        pose_cov = np.zeros((6, 6))
        pose_cov[0:2, 0:2] = self.Sigma[0:2, 0:2] # x,y
        pose_cov[5, 5] = self.Sigma[2, 2]         # yaw
        pose_cov[0, 5] = self.Sigma[0, 2]
        pose_cov[5, 0] = self.Sigma[2, 0]
        odom_msg.pose.covariance = pose_cov.flatten().tolist()
        
        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0] = self.Sigma[3, 3] # v
        twist_cov[5, 5] = self.Sigma[4, 4] # w
        twist_cov[0, 5] = self.Sigma[3, 4]
        twist_cov[5, 0] = self.Sigma[4, 3]
        odom_msg.twist.covariance = twist_cov.flatten().tolist()

        self.ekf_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFNodeTask2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()