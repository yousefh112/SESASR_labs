import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ControllerNode(Node):
    """
    A ROS 2 node that publishes velocity commands to control a robot in an expanding square pattern.
    """
    def __init__(self):
        """
        Initializes the ControllerNode.
        """
        super().__init__('controller')
        
        # Create a publisher for the /cmd_vel topic with a message type of Twist.
        # The queue size is set to 10.
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Set the frequency of publication to 1 Hz (1 message per second).
        timer_period = 1.0  # seconds
        
        # Create a timer that calls the timer_callback method at the specified frequency.
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Initialize state variables for the movement logic.
        self.n = 1          # N: duration for each movement segment, starts at 1.
        self.state = 0      # 0: +X, 1: +Y, 2: -X, 3: -Y
        self.counter = 0    # Counts seconds within the current movement segment.
        self.speed = 1.0    # Constant speed of the robot.

    def timer_callback(self):
        """
        Callback function for the timer. This function is executed at 1 Hz.
        It calculates and publishes the velocity command based on the current state.
        """
        # Create a new Twist message.
        msg = Twist()

        # Determine the linear velocity based on the current state.
        if self.state == 0:
            # Move along the positive X-axis.
            msg.linear.x = self.speed
        elif self.state == 1:
            # Move along the positive Y-axis.
            msg.linear.y = self.speed
        elif self.state == 2:
            # Move along the negative X-axis.
            msg.linear.x = -self.speed
        elif self.state == 3:
            # Move along the negative Y-axis.
            msg.linear.y = -self.speed

        # Publish the message.
        self.publisher_.publish(msg)
        
        # Log the published message content to the console.
        self.get_logger().info(f'Publishing: linear.x={msg.linear.x}, linear.y={msg.linear.y}, N={self.n}')
        
        # Increment the counter for the current movement segment.
        self.counter += 1

        # Check if the current movement segment is complete.
        if self.counter >= self.n:
            # Reset the counter and move to the next state.
            self.counter = 0
            self.state = (self.state + 1) % 4
            
            # If a full cycle (all 4 directions) is complete, increment N.
            if self.state == 0:
                self.n += 1

def main(args=None):
    """
    Main function to initialize and run the ROS 2 node.
    """
    # Initialize the rclpy library.
    rclpy.init(args=args)

    # Create an instance of the ControllerNode.
    controller_node = ControllerNode()

    # Spin the node to keep it running and processing callbacks.
    rclpy.spin(controller_node)

    # Clean up and destroy the node upon shutdown.
    controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
