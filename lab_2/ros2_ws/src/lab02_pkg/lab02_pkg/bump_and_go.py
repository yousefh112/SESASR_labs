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
    """ROS2 node that implements a bump-and-go navigation behavior with goal seeking"""
    
    def __init__(self):
        super().__init__('bump_and_go_node')

        # Declare parameters for tuning robot behavior
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_speed', 1.5)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('min_distance_threshold', 0.6)

        # Read parameters from config or use defaults
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.min_distance_threshold = self.get_parameter('min_distance_threshold').value

        # Publisher for robot velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers for sensor data (laser and odometry)
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Timer to run control loop at specified frequency
        self.timer = self.create_timer(1.0 / self.control_frequency, self.control_loop)

        # Internal state variables for tracking robot motion and position
        self.scan_data = None
        self.current_yaw = 0.0
        self.state = 'GO_FORWARD'
        self.turn_direction = 1  # 1 = left, -1 = right
        self.position = Point()
        
        # Define goal position that robot should navigate to
        self.goal_pos = Pose()
        self.goal_pos.position.x = 6.5
        self.goal_pos.position.y = 3.0

    def odom_callback(self, msg):
        """Extract robot orientation and position from odometry data"""
        quat = msg.pose.pose.orientation
        quaternion = [quat.x, quat.y, quat.z, quat.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
        self.current_yaw = yaw
        self.position = msg.pose.pose.position

    def laser_callback(self, msg):
        """Store latest laser scan data for obstacle detection"""
        self.scan_data = msg

    def normalize_angle(self, angle):
        """Normalize angle to be within [-pi, pi] range"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def control_loop(self):
        """Main control loop that handles state machine and motion planning"""
        if self.scan_data is None:
            return

        # Extract front laser ranges (±10° for narrower detection zone)
        front_indices = list(range(10)) + list(range(len(self.scan_data.ranges) - 10, len(self.scan_data.ranges)))
        valid_front = [(i, self.scan_data.ranges[i]) for i in front_indices if not np.isinf(self.scan_data.ranges[i])]
        min_index, min_front = min(valid_front, key=lambda x: x[1]) if valid_front else (None, float('inf'))
        min_angle = self.scan_data.angle_min + min_index * self.scan_data.angle_increment if min_index is not None else None

        # Calculate distance to goal
        distance_to_goal = np.sqrt(
            (self.position.x - self.goal_pos.position.x) ** 2 +
            (self.position.y - self.goal_pos.position.y) ** 2
        )
        
        # Update state based on proximity to goal
        if distance_to_goal < 0.7 and distance_to_goal > 0.4:
            self.state = 'GOAL'
            self.get_logger().info('Goal detected')
        elif distance_to_goal <= 0.4:
            self.get_logger().info('At goal position! Stopping robot.')
            self.state = 'STOP'

        twist = Twist()

        # Reduce obstacle threshold when approaching goal for finer control
        if distance_to_goal < 1.0:
            dynamic_threshold = 0.3
        else:
            dynamic_threshold = self.min_distance_threshold

        # --- STATE MACHINE: Handle robot behavior based on current state ---
        if self.state == 'GO_FORWARD':
            # Move forward; switch to rotation if obstacle detected
            if min_front < dynamic_threshold:
                self.get_logger().info('Obstacle detected! Switching to ROTATE')
                self.state = 'ROTATE'
                self.turn_direction = self.choose_turn_direction()
            else:
                twist.linear.x = self.linear_speed

        elif self.state == 'ROTATE':
            # Rotate in place until front path is clear
            if min_front > dynamic_threshold + 0.1:
                self.get_logger().info('Path clear. Switching to GO_FORWARD')
                self.state = 'GO_FORWARD'
            else:
                twist.angular.z = self.turn_direction * self.angular_speed

        elif self.state == 'GOAL':
            # Final approach to goal: align heading and move closer
            dinamic_threshold = 0
            goal_dx = self.goal_pos.position.x - self.position.x
            goal_dy = self.goal_pos.position.y - self.position.y
            goal_angle = np.arctan2(goal_dy, goal_dx)
            angle_error = self.normalize_angle(goal_angle - self.current_yaw)

            # Move toward goal while ignoring obstacles
            if distance_to_goal > 0.4:
                if abs(angle_error) > np.deg2rad(15):
                    # Turn to align with goal
                    twist.angular.z = np.sign(angle_error) * (self.angular_speed * 0.5)
                    twist.linear.x = 0.05  # Move slightly forward while turning
                else:
                    # Aligned with goal, move forward at reduced speed
                    twist.linear.x = self.linear_speed * 0.5
            else:
                # Very close to goal, stop
                self.state = 'STOP'
                self.get_logger().info('Approaching goal, preparing to stop.')
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info('Robot stopped at goal position.')

        # Publish velocity command to robot
        self.cmd_vel_pub.publish(twist)

    def choose_turn_direction(self):
        """Determine optimal rotation direction by comparing left and right clearances"""
        ranges = np.array(self.scan_data.ranges)
        left = ranges[60:100]
        right = ranges[260:300]
        left_avg = np.mean([r for r in left if not np.isinf(r)] or [float('inf')])
        right_avg = np.mean([r for r in right if not np.isinf(r)] or [float('inf')])
        return 1 if left_avg > right_avg else -1


def main(args=None):
    """Initialize and run the ROS2 node"""
    rclpy.init(args=args)
    node = BumpAndGo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()