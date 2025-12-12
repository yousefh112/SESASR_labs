import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose

class LocalizationNode(Node):
    def __init__(self):
        super().__init__('localization') #
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.publisher_ = self.create_publisher(Pose, '/pose', 10) #

        # Initialize position at the origin
        self.current_pose = Pose()
        self.current_pose.position.x = 0.0
        self.current_pose.position.y = 0.0
        self.current_pose.position.z = 0.0
        self.current_pose.orientation.w = 1.0 # Default orientation (no rotation)
        
        self.get_logger().info('Localization node started, current pose: (0.0, 0.0)')

    def cmd_vel_callback(self, msg):
        # Estimate new position based on 1s period
        # delta_x = linear_x * 1.0s
        # delta_y = linear_y * 1.0s
        self.current_pose.position.x += msg.linear.x * 1.0
        self.current_pose.position.y += msg.linear.y * 1.0
        
        # Publish the new pose
        self.publisher_.publish(self.current_pose)
        
        # Log the published message
        log_msg = (f'Publishing Pose: x={self.current_pose.position.x}, '
                   f'y={self.current_pose.position.y}')
        self.get_logger().info(log_msg) #

def main(args=None):
    rclpy.init(args=args)
    localization_node = LocalizationNode()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()