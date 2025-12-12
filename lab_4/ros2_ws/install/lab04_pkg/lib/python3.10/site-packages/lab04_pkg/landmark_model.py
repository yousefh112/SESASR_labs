import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from  matplotlib.patches import Arc

arrow = u'$\u2191$'

# --- Utility Functions (Gaussian, compute_p_hit_dist, landmark_range_bearing_sensor, etc.) ---
# These remain unchanged and are correct implementations of the sensor model components.

def gaussian(x, mu, sigma):
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)


def compute_p_hit_dist(dist, max_dist, sigma):
    # Normalize the Gaussian over [0, max_dist]
    normalize_hit = 1e-9
    for j in range(round(max_dist)):
        normalize_hit += gaussian(j, 0., sigma)
    normalize_hit = 1. / normalize_hit

    p_hit = gaussian(dist, 0., sigma)*normalize_hit

    return p_hit


def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=math.pi/2):
    """""
    Simulate the detection of a landmark with a virtual sensor able to estimate range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]

    # Add noise to true range and bearing
    r_ = math.dist([x, y], [m_x, m_y]) + np.random.normal(0., sigma[0])
    phi_ = math.atan2(m_y - y, m_x - x) - theta + np.random.normal(0., sigma[1])
    
    # Normalize bearing to [-pi, pi] for consistency
    phi_ = math.atan2(math.sin(phi_), math.cos(phi_))

    # filter z for a more realistic sensor simulation (add a max range distance and a FOV)
    if r_ > max_range or abs(phi_) > fov / 2:
        return None

    return [r_, phi_]

def landmark_model_prob(z, landmark, robot_pose, max_range, fov, sigma):
    """""
    Landmark sensor model algorithm:
    ...
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]
    sigma_r, sigma_phi = sigma[:]

    r_hat = math.dist([x, y], [m_x, m_y])
    phi_hat = math.atan2(m_y - y, m_x - x) - theta
    p = compute_p_hit_dist(z[0] - r_hat, max_range, sigma_r) * compute_p_hit_dist(z[1] - phi_hat, fov/2, sigma_phi)

    return p

def landmark_model_sample_pose(z, landmark, sigma):
    """""
    Sample a robot pose from the landmark model (Inverse Sensor Model)
    ...
    """""
    m_x, m_y = landmark[:]
    sigma_r, sigma_phi = sigma[:]

    gamma_hat = np.random.uniform(0, 2*math.pi)
    r_hat = z[0] + np.random.normal(0, sigma_r)
    phi_hat = z[1] + np.random.normal(0, sigma_phi)

    # Calculate sampled pose [x', y', theta']
    x_ = m_x + r_hat * math.cos(gamma_hat)
    y_ = m_y + r_hat * math.sin(gamma_hat)
    theta_ = gamma_hat - math.pi - phi_hat
    
    # Normalize orientation
    theta_ = math.atan2(math.sin(theta_), math.cos(theta_))

    return np.array([x_, y_, theta_])

def landmark_model_jacobian(robot_pose, landmark, eps=1e-6):
    """""
    Computes the Jacobian Ht (derivative of the predicted measurement w.r.t the state x).
    """""
    x, y, theta = robot_pose
    m_x, m_y = landmark
    dx = m_x - x
    dy = m_y - y
    r = math.sqrt(dx**2 + dy**2)
    if r < eps:
        r = eps

    Ht = np.array([
        [-dx / r, -dy / r, 0],
        [dy / (r**2), -dx / (r**2), -1]
    ])
    return Ht


def plot_sampled_poses(robot_pose, z, landmark, sigma):
    # plot samples poses
    for i in range(1000):
        x_prime = landmark_model_sample_pose(z, landmark, sigma)
        # plot robot pose (using an arrow marker for orientation)
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(x_prime[2])-90)
        plt.scatter(x_prime[0], x_prime[1], marker=rotated_marker, s=80, facecolors='none', edgecolors='b')
    
    # plot true pose
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(robot_pose[2])-90)
    plt.scatter(robot_pose[0], robot_pose[1], marker=rotated_marker, s=140, facecolors='none', edgecolors='r', label='True Pose')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Landmark Model Pose Sampling (1000 Samples)")
    plt.legend()
    plt.show()

# --- MAIN FUNCTION ---
def main():
    plt.close('all')

    # Define State and Map
    robot_pose = np.array([0., 0., math.pi/4])
    landmarks = [
                 np.array([5., 2.]),
                 np.array([-2.5, 3.]),
                 np.array([3., 1.5]),
                 np.array([4., -1.]),
                 np.array([-2., -2.])
                 ]
    
    # Sensor Parameters and Noise (sigma is the noise coefficient input)
    sigma = np.array([0.3, math.pi/24])
    fov = math.pi/3
    max_range = 6.0
    
    # Select the landmark for the single measurement
    landmark = landmarks[0] 

    # --- 1. Generate the Given Measurement z ---
    # The required measurement z = [r, phi] must be generated from the sensor model.
    z = landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=max_range, fov=fov)

    if z is None:
        print(f"Landmark at {landmark} is out of range or FOV from {robot_pose}. Cannot proceed with sampling.")
        return
        
    print(f"--- Landmark Measurement Model Simulation ---")
    print(f"Given Measurement z: [r={z[0]:.3f}, phi={z[1]:.3f} rad]")
    print(f"Landmark Position: {landmark}")
    print(f"Robot True Pose: {robot_pose}")
    print("-------------------------------------------\n")

    # --- 2. Compute the Jacobian Hx (Requirement Met) ---
    Hx = landmark_model_jacobian(robot_pose, landmark)
    print("--- Jacobian Hx (w.r.t State) at Initial Pose ---")
    print("Jacobian Hx:\n", Hx)
    print("--------------------------------------------------\n")


    # --- 3. Plot 1000 Sampled Poses (Requirement Met) ---
    # Plot the landmark used for sampling (the cube in the center)
    plt.plot(landmark[0], landmark[1], "sk", ms=10, label='Landmark M')
    
    # Plot 1000 sampled poses
    plot_sampled_poses(robot_pose, z, landmark, sigma)
    
    plt.close('all')


if __name__ == "__main__":
    main()