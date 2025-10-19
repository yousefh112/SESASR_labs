import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
import math

class ResetNode(Node):
    """
    A ROS 2 node that monitors the robot's pose and publishes a reset signal
    if the robot travels too far from the origin.
    """
    def __init__(self):
        """
        Initializes the ResetNode.
        """
        super().__init__('reset_node')
        
        # Create a subscriber to the /pose topic.
        self.subscription = self.create_subscription(
            Pose,
            '/pose',
            self.pose_callback,
            10)
            
        # Create a publisher for the /reset topic with a message type of Bool.
        self.publisher_ = self.create_publisher(Bool, '/reset', 10)
        
        # The distance from the origin at which to trigger the reset.
        self.reset_distance_threshold_ = 6.0
        
        # A flag to ensure we only send the reset signal once per excursion.
        self.reset_sent_ = False

    def pose_callback(self, msg):
        """
        Callback function for the /pose subscriber.
        Calculates distance from origin and publishes a reset signal if it exceeds the threshold.
        """
        # Extract the robot's x and y coordinates from the Pose message.
        x = msg.position.x
        y = msg.position.y
        
        # Calculate the Euclidean distance from the origin (0,0).
        distance = math.sqrt(x**2 + y**2)
        
        # Check if the distance exceeds the threshold and if we haven't already sent a reset signal.
        if distance > self.reset_distance_threshold_ and not self.reset_sent_:
            # Create a new Bool message.
            reset_msg = Bool()
            reset_msg.data = True
            
            # Publish the reset message.
            self.publisher_.publish(reset_msg)
            
            # Set the flag to True so we don't spam the topic.
            self.reset_sent_ = True
            
            # Log the event to the console.
            self.get_logger().info(
                f'Distance {distance:.2f}m exceeded threshold. Publishing reset signal.'
            )
        # This part handles re-arming the reset logic if the robot were to come back in range.
        elif distance <= self.reset_distance_threshold_ and self.reset_sent_:
            self.get_logger().info('Robot is back within the safe zone. Resetting flag.')
            self.reset_sent_ = False


def main(args=None):
    """
    Main function to initialize and run the ROS 2 node.
    """
    rclpy.init(args=args)

    reset_node = ResetNode()

    rclpy.spin(reset_node)

    reset_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
