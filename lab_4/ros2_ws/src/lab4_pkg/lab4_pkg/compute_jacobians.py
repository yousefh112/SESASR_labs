#!/usr/bin/env python3

"""
compute_jacobians.py

This script uses the 'sympy' library to symbolically compute the Jacobian
matrices required for Lab 4 (Localization with EKF), Tasks 0, 1, and 2.

- Task 0/1 Models (3-DOF state: x, y, theta):
  - G: Motion model (g) w.r.t. state (x_t)
  - V: Motion model (g) w.r.t. control (u_t)
  - H: Landmark measurement model (h) w.r.t. state (x_t)

- Task 2 Models (5-DOF state: x, y, theta, v, w):
  - G_task2: Motion model (g_task2) w.r.t. state (x_t_task2)
  - H_landmark_task2: Landmark model (h) w.r.t. state (x_t_task2)
  - H_odom: Odometry measurement model (h_odom) w.r.t. state (x_t_task2)
  - H_imu: IMU measurement model (h_imu) w.r.t. state (x_t_task2)
  
Based on the models described in the Lab 4 PDF[cite: 41, 58, 61, 99].
The use of sympy is suggested in the lab document.
"""

from sympy import symbols, Matrix, sin, cos, atan2, sqrt, init_printing

def main():
    # Use pretty printing for the console output
    init_printing(use_unicode=True)

    print("=====================================================")
    print(" Jacobians for Task 0 & 1 (State: [x, y, theta])")
    print("=====================================================")
    
    # --- Define Symbols for Task 0/1 ---
    # State variables [x, y, theta]
    x, y, theta = symbols('x y theta')
    # Control variables [v, w]
    v, w = symbols('v w')
    # Other variables
    dt = symbols('dt') # Time step

    # State vector
    x_t = Matrix([x, y, theta])
    # Control vector
    u_t = Matrix([v, w])

    # NOTE: The non-linear motion model is defined assuming w != 0
    # Your EKF code must handle the w = 0 case separately (straight-line motion).
    
    # --- 1. Motion Model g(x_t, u_t) ---
    # x' = x - (v/w)*sin(theta) + (v/w)*sin(theta + w*dt)
    # y' = y + (v/w)*cos(theta) - (v/w)*cos(theta + w*dt)
    # theta' = theta + w*dt
    
    g = Matrix([
        x - (v/w)*sin(theta) + (v/w)*sin(theta + w*dt),
        y + (v/w)*cos(theta) - (v/w)*cos(theta + w*dt),
        theta + w*dt
    ])

    # --- 2. Measurement Model h(x_t) ---
    # Landmark position (mx, my)
    mx, my = symbols('mx my')
    
    # q = (mx - x)^2 + (my - y)^2
    q_sq = (mx - x)**2 + (my - y)**2
    q = sqrt(q_sq)
    
    # r = sqrt(q)
    # phi = atan2(my - y, mx - x) - theta
    
    h = Matrix([
        q,
        atan2(my - y, mx - x) - theta
    ])

    # --- Calculate Jacobians for Task 0/1 ---
    
    # G = d(g) / d(x_t)
    G = g.jacobian(x_t)
    
    # V = d(g) / d(u_t)
    V = g.jacobian(u_t)
    
    # H = d(h) / d(x_t)
    H = h.jacobian(x_t)
    
    print("\n--- G (Motion Model w.r.t. State [x, y, theta]) ---")
    print(G)
    
    print("\n--- V (Motion Model w.r.t. Control [v, w]) ---")
    print(V)
    
    print("\n--- H (Landmark Model w.r.t. State [x, y, theta]) ---")
    print(H)

    print("\n\n=====================================================")
    print(" Jacobians for Task 2 (State: [x, y, theta, v, w])")
    print("=====================================================")

    # --- Define State for Task 2 ---
    # State vector [x, y, theta, v, w] [cite: 58]
    x_t_task2 = Matrix([x, y, theta, v, w])

    # --- 1. Motion Model g_task2(x_t_task2) ---
    # This is a constant velocity model, where the control is
    # part of the state. [cite: 61]
    g_task2 = Matrix([
        x - (v/w)*sin(theta) + (v/w)*sin(theta + w*dt), # x'
        y + (v/w)*cos(theta) - (v/w)*cos(theta + w*dt), # y'
        theta + w*dt,                                  # theta'
        v,                                             # v'
        w                                              # w'
    ])
    
    # G_task2 = d(g_task2) / d(x_t_task2)
    G_task2 = g_task2.jacobian(x_t_task2)
    
    print("\n--- G_task2 (Motion Model w.r.t. State [x, y, theta, v, w]) ---")
    print(G_task2)
    
    # --- 2. Measurement Models h_task2(x_t_task2) ---
    
    # 2a. Landmark Model [cite: 97]
    # h is the same as before [r, phi]
    H_landmark_task2 = h.jacobian(x_t_task2)
    
    print("\n--- H_landmark_task2 (Landmark Model w.r.t. State [x, y, theta, v, w]) ---")
    print(H_landmark_task2)
    
    # 2b. Odom Model [cite: 96, 99]
    # h_odom = [v, w]
    h_odom = Matrix([v, w])
    H_odom = h_odom.jacobian(x_t_task2)
    
    print("\n--- H_odom (Odom Model w.r.t. State [x, y, theta, v, w]) ---")
    print(H_odom)
    
    # 2c. IMU Model [cite: 96, 99]
    # h_imu = [w]
    h_imu = Matrix([w])
    H_imu = h_imu.jacobian(x_t_task2)
    
    print("\n--- H_imu (IMU Model w.r.t. State [x, y, theta, v, w]) ---")
    print(H_imu)
    print("\n=====================================================\n")

if __name__ == "__main__":
    main()