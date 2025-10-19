import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose

class LocalizationNode(Node):
    """
    A ROS 2 node that subscribes to velocity commands and estimates the robot's pose.
    """
    def __init__(self):
        """
        Initializes the LocalizationNode.
        """
        super().__init__('localization')
        
        # Create a subscriber to the /cmd_vel topic.
        # The callback function `cmd_vel_callback` is called for each message.
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        
        # Create a publisher for the /pose topic with a message type of Pose.
        self.publisher_ = self.create_publisher(Pose, '/pose', 10)
        
        # Initialize the robot's pose at the origin.
        self.robot_pose_ = Pose()
        self.robot_pose_.position.x = 0.0
        self.robot_pose_.position.y = 0.0
        self.robot_pose_.position.z = 0.0
        self.robot_pose_.orientation.x = 0.0
        self.robot_pose_.orientation.y = 0.0
        self.robot_pose_.orientation.z = 0.0
        self.robot_pose_.orientation.w = 1.0 # Identity quaternion for no rotation

    def cmd_vel_callback(self, msg):
        """
        Callback function for the /cmd_vel subscriber.
        Estimates the new pose based on the received velocity command.
        """
        # Time delta is 1.0 second, as the controller publishes at 1 Hz.
        dt = 1.0
        
        # Update the robot's position based on the linear velocity.
        # dx = vx * dt
        # dy = vy * dt
        self.robot_pose_.position.x += msg.linear.x * dt
        self.robot_pose_.position.y += msg.linear.y * dt
        
        # The task does not involve rotation, so orientation remains unchanged.
        
        # Publish the updated pose.
        self.publisher_.publish(self.robot_pose_)
        
        # Log the published pose to the console.
        self.get_logger().info(
            f'Publishing Pose: x={self.robot_pose_.position.x:.1f}, y={self.robot_pose_.position.y:.1f}'
        )

def main(args=None):
    """
    Main function to initialize and run the ROS 2 node.
    """
    # Initialize the rclpy library.
    rclpy.init(args=args)

    # Create an instance of the LocalizationNode.
    localization_node = LocalizationNode()

    # Spin the node to keep it running and processing callbacks.
    rclpy.spin(localization_node)

    # Clean up and destroy the node upon shutdown.
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
