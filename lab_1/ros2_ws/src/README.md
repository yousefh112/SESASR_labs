
### **Methodology Report for Lab 01: ROS 2 Basic Concepts**

**Date:** October , 2025
**Author:** Gemini AI

#### **1. Introduction**

The primary objective of this lab was to understand and implement the fundamental concepts of the Robot Operating System 2 (ROS 2). The tasks involved creating a ROS 2 package, developing communicating nodes using the publisher/subscriber pattern, managing node state, and preparing for data visualization. This report details the methodology and architectural decisions made to solve each task systematically.

The core environment used was **ROS 2 Humble** with the **Python (`rclpy`)** client library. All development was contained within a `colcon` workspace, following standard ROS 2 practices.

-----

#### **2. Task-by-Task Implementation Methodology**

##### **Task 1: Package Creation**

The foundation of the project was established by creating a ROS 2 package named `lab01_pkg`. This was accomplished using the standard command-line tool:

```bash
ros2 pkg create --build-type ament_python lab01_pkg
```

This command scaffolded a standard Python package structure, including the essential configuration files:

  * **`package.xml`**: Contains meta-information about the package, such as its name, version, and dependencies.
  * **`setup.py`**: A Python script used by the `colcon` build system to understand how to install the package and, crucially, where to find its executables (nodes).
  * **`setup.cfg`**: Configuration for the setup tools.

This structured approach ensures that the package is discoverable and buildable within the ROS 2 ecosystem.

##### **Task 2: The `controller` Node (Publisher)**

The goal of this task was to create a node that publishes velocity commands (`geometry_msgs/msg/Twist`) to the `/cmd_vel` topic at a fixed frequency of 1 Hz.

  * **Node Architecture:** A Python class, `ControllerNode`, was created inheriting from `rclpy.node.Node`. This object-oriented approach encapsulates all logic and ROS 2 interfaces related to the controller.
  * **Publisher Creation:** Inside the node's `__init__` method, a publisher was instantiated using `self.create_publisher(Twist, '/cmd_vel', 10)`. The message type `Twist` and topic name `/cmd_vel` were specified as per the task requirements.
  * **Timed Logic and State Machine:** To publish at a consistent 1 Hz rate, a `rclpy.Timer` was used. This timer calls a `timer_callback` method every 1.0 second. A simple state machine was implemented using class member variables (`self.state`, `self.counter`, `self.N`) to manage the movement pattern:
    1.  `self.state`: An integer from 0 to 3, representing the current movement axis (+X, +Y, -X, -Y).
    2.  `self.counter`: Counts the seconds elapsed in the current state.
    3.  `self.N`: The duration in seconds for each movement direction in the current cycle.
        After each full cycle (all four states completed), `N` is incremented.
  * **Logging:** As required, `self.get_logger().info()` was used within the callback to print the content of each published message to the console for real-time monitoring.
  * **Executable Entry Point:** The node was registered as a console script in `setup.py` under `entry_points`, mapping the executable name `controller` to the `main` function of the script.

##### **Task 3: The `localization` Node (Subscriber)**

This task required creating a node to subscribe to the `/cmd_vel` topic, estimate the robot's position, and publish it as a `geometry_msgs/msg/Pose` message on the `/pose` topic.

  * **Subscriber Architecture:** A `LocalizationNode` class was created. It was configured with a subscriber using `self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)`.
  * **Callback-Driven Logic:** The core logic resides in the `cmd_vel_callback` method, which is executed automatically whenever a message arrives on the `/cmd_vel` topic.
  * **Position Estimation:** A simple kinematic integration was performed. The node maintains the robot's current position in a `self.current_pose` variable. Inside the callback, the position is updated using the formula:
    `new_position = old_position + velocity * delta_time`
    Given the controller's 1 Hz frequency, `delta_time` was assumed to be 1.0 second.
  * **Publisher Creation:** A publisher was created to broadcast the updated `self.current_pose` on the `/pose` topic. This allows other nodes to access the robot's estimated position.
  * **Executable Entry Point:** Similar to the controller, this node was registered in `setup.py`.

##### **Task 5: The `reset_node` (Supervisor)**

This task introduced a supervisory node to monitor the system and react to a specific condition.

  * **Hybrid Subscriber/Publisher:** The `ResetNode` subscribes to the `/pose` topic to monitor the robot's position.
  * **Conditional Logic:** In its `pose_callback` function, it calculates the robot's Euclidean distance from the origin (`sqrt(x^2 + y^2)`). An `if` statement checks if this distance exceeds the specified threshold of 6.0 meters.
  * **Event Publishing:** If the condition is met, the node publishes a `std_msgs/msg/Bool` message with a value of `True` to a new topic, `/reset`. This acts as a global signal for other nodes to reset their states.

##### **Task 6: Resettable Nodes (`controller_reset` & `localization_reset`)**

The final coding task was to modify the initial nodes to respond to the `/reset` signal.

  * **Code Re-use:** The existing `controller_node.py` and `localization_node.py` files were copied to create `controller_reset_node.py` and `localization_reset_node.py`.
  * **Adding a Reset Subscriber:** In both new nodes, a subscription to the `/reset` topic was added. This subscription is linked to a new `reset_callback` method.
  * **State Reset Logic:**
      * In `ControllerResetNode`, the `reset_callback` function resets the state machine variables (`self.N`, `self.counter`, `self.state`) to their initial values (`1`, `0`, `0`, respectively).
      * In `LocalizationResetNode`, the `reset_callback` resets the `self.current_pose` position coordinates back to the origin (0.0, 0.0).
  * **Encapsulation:** The reset logic was encapsulated in dedicated methods (`reset_state()` and `reset_pose()`) for clarity and re-use. These methods were also called during the initial `__init__` process to ensure the nodes start in a clean state.
  * **Logging:** Log messages were added to the reset callbacks to provide explicit confirmation that a reset event was received and handled.

#### **3. Visualization and System Analysis (Tasks 4 & 7)**

While not involving code generation, the methodology to complete these tasks is as follows:

  * **`rqt_graph`**: This tool is used to visualize the computational graph of the running ROS 2 system. By running `ros2 run rqt_graph rqt_graph`, a diagram is generated showing the active nodes and the topics that connect them.  This is a critical step to verify that the nodes are communicating as intended (e.g., `controller_reset` -\> `/cmd_vel` -\> `localization_reset` -\> `/pose` -\> `reset_node` -\> `/reset`).
  * **`plotjuggler`**: This tool is used for real-time data plotting. The procedure involves launching `plotjuggler`, connecting to the ROS 2 topics, and dragging specific message fields (e.g., `/cmd_vel/linear/x`, `/pose/position/y`) onto a timeline graph. This allows for direct verification of the system's behavior, confirming that the square-wave velocity input from the controller correctly produces a ramping position output from the localizer, which then resets as expected.

-----

#### **4. Conclusion**

The methodology employed a step-by-step, modular approach. Each task built upon the previous one, starting with the package structure, followed by the creation of individual nodes with distinct responsibilities (control, estimation, supervision), and finally integrating them into a cohesive system. The use of standard ROS 2 tools and patterns ensures that the resulting software is maintainable, debuggable, and extensible. The successful completion of all tasks demonstrates a practical understanding of core ROS 2 principles.