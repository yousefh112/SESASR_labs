import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool

class LocalizationResetNode(Node):
    """
    A ROS 2 node that estimates robot position based on /cmd_vel and
    resets its position upon receiving a signal on the /reset topic.
    """
    def __init__(self):
        """
        Initializes the LocalizationResetNode.
        """
        super().__init__('localization_reset')
        
        # Subscriber for velocity commands.
        self.cmd_vel_subscription_ = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
            
        # Subscriber for the reset signal.
        self.reset_subscription_ = self.create_subscription(
            Bool,
            '/reset',
            self.reset_callback,
            10)
        
        # Publisher for the estimated pose.
        self.publisher_ = self.create_publisher(Pose, '/pose', 10)
        
        # Initialize the robot's pose at the origin.
        self.robot_pose_ = Pose()
        
        self.get_logger().info('Localization Reset Node has started.')

    def cmd_vel_callback(self, msg):
        """
        Callback function for the /cmd_vel subscriber.
        Estimates the new pose and publishes it.
        """
        dt = 1.0  # Time step is 1 second, as controller publishes at 1 Hz.
        
        # Update the robot's pose based on velocity.
        self.robot_pose_.position.x += msg.linear.x * dt
        self.robot_pose_.position.y += msg.linear.y * dt
        
        # Publish the updated pose.
        self.publisher_.publish(self.robot_pose_)
        self.get_logger().info(f'Publishing Pose: x={self.robot_pose_.position.x:.1f}, y={self.robot_pose_.position.y:.1f}')
        
    def reset_callback(self, msg):
        """
        Callback function for the /reset subscriber.
        Resets the robot's pose to the origin.
        """
        if msg.data:
            self.get_logger().info('RESET signal received. Resetting pose to origin.')
            # Create a new, empty Pose message, which defaults to the origin.
            self.robot_pose_ = Pose()

def main(args=None):
    """
    Main function to initialize and run the ROS 2 node.
    """
    rclpy.init(args=args)
    localization_reset_node = LocalizationResetNode()
    rclpy.spin(localization_reset_node)
    localization_reset_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
