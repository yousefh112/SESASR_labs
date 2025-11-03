import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf_transformations
import numpy as np
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point


class BumpAndGo(Node):
    """
    A ROS2 node that implements a bump-and-go navigation behavior.
    The robot moves forward, detects obstacles with a laser, rotates around them,
    and navigates toward a goal position.
    """

    def __init__(self):
        super().__init__('bump_and_go_node')

        # ============ Parameter Declarations ============
        # Core motion parameters
        self.declare_parameter('linear_speed', 0.2)          # Forward velocity (m/s)
        self.declare_parameter('angular_speed', 1.5)         # Rotation velocity (rad/s)
        self.declare_parameter('control_frequency', 10.0)    # Loop frequency (Hz)
        self.declare_parameter('min_distance_threshold', 0.6) # Obstacle detection distance (m)
        
        # Sensor and perception parameters
        self.declare_parameter('frontal_cone_deg', 20.0)     # Frontal detection cone width (degrees)
        self.declare_parameter('odom_topic', '/odom')        # Odometry topic (/odom or /ground_truth)
        
        # Goal-reaching parameters
        self.declare_parameter('goal_x', 6.5)                # Goal X coordinate (m)
        self.declare_parameter('goal_y', 3.0)                # Goal Y coordinate (m)
        self.declare_parameter('goal_approach_dist', 0.7)    # Distance to switch to final approach (m)
        self.declare_parameter('goal_tolerance', 0.4)        # Distance to consider goal reached (m)

        # ============ Read Parameters ============
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.min_distance_threshold = self.get_parameter('min_distance_threshold').value
        self.frontal_cone_deg = self.get_parameter('frontal_cone_deg').value
        odom_topic = self.get_parameter('odom_topic').value
        goal_x = self.get_parameter('goal_x').value
        goal_y = self.get_parameter('goal_y').value
        self.goal_approach_dist = self.get_parameter('goal_approach_dist').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value

        # ============ Publishers & Subscribers ============
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        # ============ Control Timer ============
        self.timer = self.create_timer(1.0 / self.control_frequency, self.control_loop)

        # ============ Internal State Variables ============
        self.scan_data = None              # Latest laser scan data
        self.current_yaw = 0.0             # Current robot heading (radians)
        self.position = Point()            # Current robot position
        self.state = 'GO_FORWARD'          # State machine: GO_FORWARD, ROTATE, GOAL, STOP
        self.turn_direction = 1            # Rotation direction: 1 (left), -1 (right)
        
        # ============ Goal Configuration ============
        self.goal_pos = Pose()
        self.goal_pos.position.x = goal_x
        self.goal_pos.position.y = goal_y

    def odom_callback(self, msg):
        """Extract robot pose (position and yaw) from odometry message."""
        quat = msg.pose.pose.orientation
        quaternion = [quat.x, quat.y, quat.z, quat.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
        self.current_yaw = yaw
        self.position = msg.pose.pose.position

    def laser_callback(self, msg):
        """Store the latest laser scan data."""
        self.scan_data = msg

    def normalize_angle(self, angle):
        """Normalize angle to range [-π, π]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def control_loop(self):
        """Main control loop executed at fixed frequency."""
        if self.scan_data is None or self.scan_data.angle_increment == 0:
            return

        # ============ Obstacle Detection (Frontal Cone) ============
        # Calculate which scan indices fall within the frontal detection cone
        num_ranges = len(self.scan_data.ranges)
        cone_half_angle_rad = np.deg2rad(self.frontal_cone_deg / 2.0)
        indices_per_half = int(cone_half_angle_rad / self.scan_data.angle_increment)

        # Handle wrap-around for 360-degree scan (front = index 0)
        front_indices = list(range(indices_per_half + 1)) + \
                        list(range(num_ranges - indices_per_half, num_ranges))

        # Filter valid ranges (exclude inf and nan values)
        valid_front_ranges = [self.scan_data.ranges[i] for i in front_indices 
                              if not np.isinf(self.scan_data.ranges[i]) and not np.isnan(self.scan_data.ranges[i])]

        # Use mean for noise robustness
        front_distance = np.mean(valid_front_ranges) if valid_front_ranges else float('inf')

        # ============ Distance to Goal Calculation ============
        distance_to_goal = np.sqrt(
            (self.position.x - self.goal_pos.position.x) ** 2 +
            (self.position.y - self.goal_pos.position.y) ** 2
        )

        # ============ Goal State Transitions ============
        if self.state != 'STOP':
            if distance_to_goal <= self.goal_tolerance:
                # Goal reached: stop the robot
                self.get_logger().info('At goal position! Stopping robot.')
                self.state = 'STOP'
            elif self.state != 'GOAL' and distance_to_goal < self.goal_approach_dist:
                # Close enough to goal: switch to final approach mode
                self.get_logger().info('Goal detected, switching to final approach.')
                self.state = 'GOAL'

        # ============ Command Velocity Initialization ============
        twist = Twist()

        # ============ Adaptive Obstacle Threshold ============
        # Use stricter threshold when close to goal
        dynamic_threshold = 0.3 if distance_to_goal < 1.0 else self.min_distance_threshold

        # ============ STATE MACHINE ============
        if self.state == 'GO_FORWARD':
            # Move forward unless obstacle detected
            if front_distance < dynamic_threshold:
                self.get_logger().info('Obstacle detected! Switching to ROTATE')
                self.state = 'ROTATE'
                self.turn_direction = self.choose_turn_direction()
            else:
                twist.linear.x = self.linear_speed

        elif self.state == 'ROTATE':
            # Rotate until path is clear
            if front_distance > dynamic_threshold + 0.1:
                self.get_logger().info('Path clear. Switching to GO_FORWARD')
                self.state = 'GO_FORWARD'
            else:
                twist.angular.z = self.turn_direction * self.angular_speed

        elif self.state == 'GOAL':
            # Final approach: align with goal and move toward it
            goal_dx = self.goal_pos.position.x - self.position.x
            goal_dy = self.goal_pos.position.y - self.position.y
            goal_angle = np.arctan2(goal_dy, goal_dx)
            angle_error = self.normalize_angle(goal_angle - self.current_yaw)

            if distance_to_goal > self.goal_tolerance:
                if abs(angle_error) > np.deg2rad(15):
                    # Turn to face goal while creeping forward
                    twist.angular.z = np.sign(angle_error) * (self.angular_speed * 0.5)
                    twist.linear.x = 0.05
                else:
                    # Aligned with goal: move forward slowly
                    twist.linear.x = self.linear_speed * 0.5
            else:
                # Goal reached
                self.state = 'STOP'
                self.get_logger().info('Approaching goal, preparing to stop.')
        
        elif self.state == 'STOP':
            # Stop all motion
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        # ============ Publish Command ============
        self.cmd_vel_pub.publish(twist)

    def choose_turn_direction(self):
        """
        Decide rotation direction by comparing left and right space.
        Returns 1 for left turn, -1 for right turn.
        """
        ranges = np.array(self.scan_data.ranges)
        left = ranges[60:100]
        right = ranges[260:300]
        
        # Calculate average distance on each side (ignore inf values)
        left_avg = np.mean([r for r in left if not np.isinf(r)] or [float('inf')])
        right_avg = np.mean([r for r in right if not np.isinf(r)] or [float('inf')])
        
        # Turn toward the side with more space
        return 1 if left_avg > right_avg else -1


def main(args=None):
    """Main entry point: initialize and run the ROS2 node."""
    rclpy.init(args=args)
    node = BumpAndGo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, stopping robot.')
    finally:
        # ============ Safety Shutdown ============
        # Send zero velocity to ensure robot stops
        stop_twist = Twist()
        node.cmd_vel_pub.publish(stop_twist)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()