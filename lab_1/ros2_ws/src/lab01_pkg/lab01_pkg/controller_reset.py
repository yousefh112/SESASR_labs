import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool # Import Bool for reset subscriber

class ControllerResetNode(Node):
    def __init__(self):
        super().__init__('controller_reset')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Reset subscription
        self.reset_subscription = self.create_subscription(
            Bool,
            '/reset',
            self.reset_callback,
            10)
            
        timer_period = 1.0  # 1 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.velocity = 1.0  # Moves at 1 m/s
        self.reset_state() # Call reset_state to initialize variables

    def reset_state(self):
        # This function contains the reset logic
        self.N = 1         # Reset N to 1
        self.counter = 0   # Reset counter
        self.state = 0     # Reset state to 0 (+X)

    def reset_callback(self, msg):
        if msg.data:
            self.get_logger().info('Reset signal received. Resetting controller state.')
            self.reset_state() # Reset N, counter, and state

    def timer_callback(self):
        msg = Twist()
        
        if self.state == 0:    # N seconds along X-axis
            msg.linear.x = self.velocity
        elif self.state == 1:  # N seconds along Y-axis
            msg.linear.y = self.velocity
        elif self.state == 2:  # N seconds opposite X-axis
            msg.linear.x = -self.velocity
        elif self.state == 3:  # N seconds opposite Y-axis
            msg.linear.y = -self.velocity
            
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: N={self.N}, state={self.state}, linear_x: {msg.linear.x}, linear_y: {msg.linear.y}')
        
        self.counter += 1
        
        if self.counter >= self.N:
            self.counter = 0
            self.state = (self.state + 1) % 4
            
            if self.state == 0:
                self.N += 1

def main(args=None):
    rclpy.init(args=args)
    controller_reset_node = ControllerResetNode()
    rclpy.spin(controller_reset_node)
    controller_reset_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()