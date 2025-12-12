# Lab 3 


---

## Overview
This repository implements goal-oriented autonomous navigation and obstacle avoidance for a TurtleBot3 using ROS 2. The project contains two ROS 2 Python nodes:

- `bump_and_go_node` — simulation (goal-seeking + obstacle avoidance, 4-state FSM)  
- `bump_go_real` — real robot (robust obstacle avoidance, 2-state FSM)

Both nodes are implemented in a `BumpAndGo` class that handles sensing, computation and control.

---

## Objectives
- Design a behavior-based controller using a finite-state machine (FSM).  
- Subscribe to `/scan` (sensor_msgs/LaserScan) and `/odom` (nav_msgs/Odometry).  
- Publish to `/cmd_vel` (geometry_msgs/Twist).  
- Run a high-frequency control loop using a timer.  
- Use ROS 2 parameters (YAML) to tune speeds and thresholds.  
- Validate in Gazebo simulation and on the real TurtleBot3.

---

## Repository structure (important files)
- `bump_and_go.py` — simulation node (4-state FSM)  
- `bump_go_real.py` — real-robot node (2-state FSM)  
- `bump_and_go_params.yaml` — tunable parameters (linear_speed, angular_speed, control_frequency, min_distance_threshold, ... )  
- `lab02.launch.py` — launch file for simulation  
- `README.md` — this file

---

## Common ROS 2 structure
- Nodes: `bump_and_go_node` (simulation), `bump_go_real` (real robot)  
- Publishers: `/cmd_vel` (geometry_msgs/Twist)  
- Subscribers:
    - `/scan` (sensor_msgs/LaserScan) — real robot node uses `qos_profile_sensor_data`  
    - `/odom` (nav_msgs/Odometry) — used for goal-seeking in simulation  
- Parameters loaded from `bump_and_go_params.yaml`

---

## FSM design

### Simulation node: 4-state FSM (`bump_and_go.py`)
States: `GO_FORWARD`, `ROTATE`, `GOAL`, `STOP`

- GO_FORWARD
    - Move forward at `linear_speed`.
    - Monitor a narrow frontal cone (±10°). If an obstacle is closer than a dynamic threshold, switch to `ROTATE`.
    - Threshold reduces when near the goal for finer control.

- ROTATE
    - Rotate in place at `angular_speed`.
    - `choose_turn_direction()` compares averaged left/right sensor cones to pick the direction with more free space.
    - Return to `GO_FORWARD` when frontal path clears.

- GOAL
    - Entered when distance to goal < 0.7 m.
    - Compute heading error, align, then move slowly toward the goal. Bypass `ROTATE` logic in this state.
    - Transition to `STOP` when distance < 0.4 m.

- STOP
    - Publish zero velocity and halt.

(Figures in original report illustrate the control loop and `choose_turn_direction` logic.)

### Real-robot node: 2-state FSM (`bump_go_real.py`)
States: `GO_FORWARD`, `ROTATE`

- GO_FORWARD
    - Move forward at `linear_speed`.
    - Monitor a frontal cone (±15°). If obstacle < `min_distance_threshold`, switch to `ROTATE`.

- ROTATE
    - Rotate in place and use `choose_turn_direction()` to find free space.
    - Return to `GO_FORWARD` when frontal path clears (hysteresis: threshold + 0.1 m to prevent oscillation).

This produces continuous bump-and-go exploration with robust collision avoidance.

---

## Parameters and tuning
Key parameters (examples found in `bump_and_go_params.yaml`):
- `linear_speed` — forward speed  
- `angular_speed` — rotation speed  
- `control_frequency` — control loop rate  
- `min_distance_threshold` — obstacle detection distance  
- `dynamic_threshold` — reduced threshold near goal  
- `goal_pos` — (x, y) goal for simulation

Tuning summary (final robust set):
- `linear_speed ≈ 0.2 m/s`  
- `angular_speed ≈ 1.5 rad/s`  
- `min_distance_threshold ≈ 0.6 m` (reduced to 0.3 m near goal)  
- GOAL trigger: 0.7 m, STOP: 0.4 m

---

## Experiments

### Simulation
- Environment: Gazebo maze launched with `lab02.launch.py`.  
- Goal: `(x=6.5, y=3.0)`.  
- Method: Iterative tuning using `bump_and_go_params.yaml` and PlotJuggler for `/odom` and `/cmd_vel` visualization.  
- Result: Final parameter set allowed consistent goal navigation and successful trials across machines.

### Real robot
- Deployed `bump_go_real.py` on physical TurtleBot3 in lab maze.  
- Recorded bags (`/scan`, `/odom`, `/cmd_vel`) with `ros2 bag record`.  
- Replayed and analyzed in PlotJuggler.  
- Result: Successful runs demonstrating reliable obstacle avoidance and `choose_turn_direction` behavior; selected best run for analysis.

---

## Results summary
- Simulation: 4-state FSM reliably navigates to the goal and transitions to `STOP`. Trajectory from `/odom` matches the expected path to `(6.5, 3.0)`.  
- Real robot: 2-state FSM achieves continuous exploration without collisions; minor trajectory jitter due to sensor noise and narrow frontal cone (±15°), which improves robustness.

---

## How to run (basic)
1. Source ROS 2 and workspace:
     - source your ROS 2 setup and workspace (example)
         ```
         source /opt/ros/<distro>/setup.bash
         source install/setup.bash
         ```
2. Launch simulation (example):
     - ```
         ros2 launch <package> lab02.launch.py
         ```
     - Run node with parameters:
         ```
         ros2 run <package> bump_and_go_node --ros-args --params-file bump_and_go_params.yaml
         ```
3. Run on real robot:
     - Ensure network and QoS settings are appropriate, then:
         ```
         ros2 run <package> bump_go_real --ros-args --params-file bump_and_go_params.yaml
         ```
4. Record bags for analysis:
     - ```
         ros2 bag record /scan /odom /cmd_vel
         ```

Adjust package and file names as needed.

---

## Notes and future work
- Consider dynamic obstacle handling and path smoothing for real-world deployments.  
- Integrate additional sensors or SLAM for global navigation and dynamic goal selection.  
- Improve parameter auto-tuning and add unit/integration tests for the FSM.

---