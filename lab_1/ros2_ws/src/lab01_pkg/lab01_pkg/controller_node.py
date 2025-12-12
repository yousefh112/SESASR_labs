import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller') # 
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10) # 
        timer_period = 1.0  # 1 Hz frequency 
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.N = 1         # N starts from 1 [cite: 86]
        self.counter = 0   # Counts seconds within a state
        self.state = 0     # 0: +X, 1: +Y, 2: -X, 3: -Y
        self.velocity = 1.0  # Moves at 1 m/s [cite: 81]

    def timer_callback(self):
        msg = Twist()
        
        if self.state == 0:  # N seconds along X-axis [cite: 82]
            msg.linear.x = self.velocity
        elif self.state == 1:  # N seconds along Y-axis [cite: 83]
            msg.linear.y = self.velocity
        elif self.state == 2:  # N seconds opposite X-axis [cite: 84]
            msg.linear.x = -self.velocity
        elif self.state == 3:  # N seconds opposite Y-axis [cite: 85]
            msg.linear.y = -self.velocity
            
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: linear_x: {msg.linear.x}, linear_y: {msg.linear.y}') # 
        
        self.counter += 1
        
        # Check if N seconds have passed for the current state
        if self.counter >= self.N:
            self.counter = 0
            self.state = (self.state + 1) % 4 # Move to the next state
            
            # If we completed a full cycle (back to state 0), increment N
            if self.state == 0:
                self.N += 1  # [cite: 86]

def main(args=None):
    rclpy.init(args=args)
    controller_node = ControllerNode()
    rclpy.spin(controller_node)
    controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()