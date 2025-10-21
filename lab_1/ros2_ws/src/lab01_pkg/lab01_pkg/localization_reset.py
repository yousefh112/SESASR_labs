import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool # Import Bool for reset subscriber

class LocalizationResetNode(Node):
    def __init__(self):
        super().__init__('localization_reset')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.publisher_ = self.create_publisher(Pose, '/pose', 10)

        # Reset subscription
        self.reset_subscription = self.create_subscription(
            Bool,
            '/reset',
            self.reset_callback,
            10)

        self.current_pose = Pose()
        self.reset_pose() # Call reset_pose to initialize position
        
        self.get_logger().info('Localization (reset) node started.')

    def reset_pose(self):
        # This function resets the position to the origin
        self.current_pose.position.x = 0.0
        self.current_pose.position.y = 0.0
        self.current_pose.position.z = 0.0
        # We don't need to reset orientation, but it's good practice
        self.current_pose.orientation.w = 1.0
        self.current_pose.orientation.x = 0.0
        self.current_pose.orientation.y = 0.0
        self.current_pose.orientation.z = 0.0
        
    def reset_callback(self, msg):
        if msg.data:
            self.get_logger().info('Reset signal received. Resetting pose to origin.')
            self.reset_pose() # Reset the pose

    def cmd_vel_callback(self, msg):
        # Estimate new position (delta = v * 1.0s)
        self.current_pose.position.x += msg.linear.x * 1.0
        self.current_pose.position.y += msg.linear.y * 1.0
        
        self.publisher_.publish(self.current_pose)
        
        log_msg = (f'Publishing Pose: x={self.current_pose.position.x:.1f}, '
                   f'y={self.current_pose.position.y:.1f}')
        self.get_logger().info(log_msg)

def main(args=None):
    rclpy.init(args=args)
    localization_reset_node = LocalizationResetNode()
    rclpy.spin(localization_reset_node)
    localization_reset_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()