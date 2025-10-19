import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class ControllerResetNode(Node):
    """
    A ROS 2 node that controls a robot's movement in an expanding square pattern
    and resets its state upon receiving a signal on the /reset topic.
    """
    def __init__(self):
        """
        Initializes the ControllerResetNode.
        """
        super().__init__('controller_reset')
        
        # Publisher for velocity commands.
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for the reset signal.
        self.reset_subscription_ = self.create_subscription(
            Bool,
            '/reset',
            self.reset_callback,
            10)
            
        # Timer to publish commands at 1 Hz.
        self.timer_ = self.create_timer(1.0, self.timer_callback)
        
        # Initialize control state variables.
        self.N_ = 1
        self.state_ = 0  # 0: +X, 1: +Y, 2: -X, 3: -Y
        self.counter_ = 0
        
        self.get_logger().info('Controller Reset Node has started.')

    def timer_callback(self):
        """
        Called by the timer to publish the next velocity command.
        """
        msg = Twist()

        # Determine velocity based on the current state.
        if self.state_ == 0:  # Move along +X axis
            msg.linear.x = 1.0
        elif self.state_ == 1:  # Move along +Y axis
            msg.linear.y = 1.0
        elif self.state_ == 2:  # Move along -X axis
            msg.linear.x = -1.0
        elif self.state_ == 3:  # Move along -Y axis
            msg.linear.y = -1.0

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing cmd_vel: linear.x={msg.linear.x}, linear.y={msg.linear.y}')

        self.counter_ += 1

        # Check if it's time to change state.
        if self.counter_ >= self.N_:
            self.counter_ = 0
            self.state_ = (self.state_ + 1) % 4
            # If a full cycle is complete, increment N.
            if self.state_ == 0:
                self.N_ += 1
                self.get_logger().info(f'--- Increasing N to {self.N_} ---')
                
    def reset_callback(self, msg):
        """
        Callback function for the /reset subscriber.
        Resets the controller's state variables.
        """
        if msg.data:
            self.get_logger().info('RESET signal received. Resetting controller state.')
            self.N_ = 1
            self.state_ = 0
            self.counter_ = 0

def main(args=None):
    """
    Main function to initialize and run the ROS 2 node.
    """
    rclpy.init(args=args)
    controller_reset_node = ControllerResetNode()
    rclpy.spin(controller_reset_node)
    controller_reset_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
