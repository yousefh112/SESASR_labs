import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
import math # Import math for sqrt

class ResetNode(Node):
    def __init__(self):
        super().__init__('reset_node')
        self.subscription = self.create_subscription(
            Pose,
            '/pose',
            self.pose_callback,
            10)
        self.publisher_ = self.create_publisher(Bool, '/reset', 10)
        self.reset_distance_threshold = 6.0
        self.get_logger().info('Reset node started. Monitoring pose.')

    def pose_callback(self, msg):
        # Calculate distance from origin (sqrt(x^2 + y^2))
        x = msg.position.x
        y = msg.position.y
        distance = math.sqrt(x**2 + y**2)

        # Check if distance is larger than 6.0 m
        if distance > self.reset_distance_threshold:
            self.get_logger().warn(f'Distance {distance:.2f} > {self.reset_distance_threshold}. Publishing reset signal.')
            
            reset_msg = Bool()
            reset_msg.data = True # Publish True on /reset
            self.publisher_.publish(reset_msg)

def main(args=None):
    rclpy.init(args=args)
    reset_node = ResetNode()
    rclpy.spin(reset_node)
    reset_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()