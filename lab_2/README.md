# Lab 02 Report: ROS 2 Simulation and "Bump & Go" Controller

**Student:** [Your Name]
**Date:** November 3, 2025

## 1. Introduction and Objectives

This report details the work completed for Lab 02, "Simulation in Gazebo and visualization with Rviz2." The primary objectives of this lab were to:

* Understand and utilize core ROS 2 concepts, including parameters and launch files.
* Gain familiarity with the Gazebo simulation environment for robotics.
* Use Rviz 2 to visualize robot data, such as sensor feeds and transforms.
* Design and implement a "Bump & Go" controller to autonomously navigate a TurtleBot3 robot through a walled environment to a predefined goal.

This report focuses on the design and implementation of the final controller, as captured in the `bump_and_go.py` script.

## 2. Methodology and Implementation

The exercise required creating a ROS 2 node to navigate a TurtleBot3 Burger from a starting position `(0, 0)` to a goal (a green cube) located at approximately `(6.5, 3.0)`. The controller must rely on `/scan` (LiDAR) data for obstacle avoidance and `/odom` data for localization and heading.

The implemented solution, `bump_and_go.py`, is a Python-based ROS 2 node that operates as a finite state machine.

### 2.1. Controller State Machine

The controller logic is built around four distinct states:

1.  **`GO_FORWARD`**: This is the default state. The robot moves forward at a constant linear velocity. It continuously monitors its frontal LiDAR-based obstacle detector.
2.  **`ROTATE`**: This state is triggered when an obstacle is detected within a dynamic threshold in the `GO_FORWARD` state. Instead of rotating indefinitely, the controller implements a fixed **90-degree turn**.
3.  **`GOAL`**: This state is triggered when the robot's position is within the `goal_approach_dist` (0.7m) of the target. This state overrides obstacle avoidance for a final "parking" maneuver.
4.  **`STOP`**: This state is triggered when the robot is within the final `goal_tolerance` (0.4m) of the target, at which point it ceases all movement.

### 2.2. Key Implementation Details

The controller script (`bump_and_go.py`) includes several key features for robustness and configurability.

#### Parameterization

As specified in the lab requirements, the node is fully configurable via ROS 2 parameters. All key values are declared in the `__init__` method and read from an external parameter file. This includes:

* **Motion**: `linear_speed`, `angular_speed`
* **Sensing**: `min_distance_threshold`, `frontal_cone_deg`, `odom_topic`
* **Goal**: `goal_x`, `goal_y`, `goal_approach_dist`, `goal_tolerance`

This allows for easy tuning (e.g., switching between `/odom` and `/ground_truth`) without modifying the Python code.

#### Robust Obstacle Detection

To handle sensor noise (as mentioned in the lab hints), the obstacle detector does not rely on a single minimum value. Instead, it calculates the **mean distance** within a "frontal cone" defined by the `frontal_cone_deg` parameter. This prevents false positives from stray LiDAR readings. The detection logic is as follows:

1.  Calculate the LiDAR scan indices corresponding to the `frontal_cone_deg` (e.g., ±10°).
2.  Filter this list to include only valid, finite ranges (excluding `inf` and `nan`).
3.  Calculate the `np.mean()` of these valid ranges.
4.  Compare this `front_distance` to an adaptive threshold (which is stricter when near the goal).

#### 90-Degree Turn Logic

When an obstacle is detected, the `ROTATE` state is entered. This state performs a calculated 90-degree turn.

1.  **Choose Direction**: The `choose_turn_direction()` method first determines the best direction (left or right) by comparing the mean free space in 40-degree sectors to the left (`[60:100]`) and right (`[260:300]`).
2.  **Set Target**: The controller records its `current_yaw` and calculates a `target_yaw` by adding 90 degrees (`np.pi / 2.0`) in the chosen direction.
3.  **Execute Turn**: The robot rotates at `angular_speed` until its `current_yaw` is within a 5-degree tolerance of the `target_yaw`.
4.  **Return to `GO_FORWARD`**: Once the turn is complete, the state switches back to `GO_FORWARD`.

This fixed-turn approach is more predictable than rotating until the path is clear. If the path is still blocked, the controller will simply execute another 90-degree turn.

#### Final Goal Approach

When the `GOAL` state is triggered, the controller focuses exclusively on reaching the goal coordinates. It calculates the angle to the goal and turns to face it. It then creeps forward at a reduced speed, continuously correcting its heading, until it reaches the final `goal_tolerance` and enters the `STOP` state.

## 3. Configuration and Launching

The system was designed to be run using ROS 2 launch files, as per the lab objectives.

* **Parameter File (`bump_and_go_params.yaml`)**: A YAML file was created to store all parameter values.
* **Launch File (`start_lab02.launch.py`)**: A "master" launch file was created to:
    1.  Include the `turtlebot3_gazebo` launch file (`lab02.launch.py`) to start the simulation.
    2.  Start the `bump_and_go_node`, automatically loading all its parameters from the YAML file.

This setup allows the entire lab exercise to be started with a single `ros2 launch` command.

## 4. Conclusion

The implemented "Bump & Go" controller successfully meets all requirements of the lab exercise. The robot reliably navigates the simulated environment, uses its LiDAR to detect walls, and executes clean 90-degree turns to avoid them. By using a state machine and robust sensing logic (mean distance), the controller avoids getting stuck. Finally, the goal-seeking state allows it to successfully identify and stop at the target cube.