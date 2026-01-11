import numpy as np
import math
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# --- METRICS FUNCTIONS ---

def mse(actual: np.ndarray, predicted: np.ndarray):
    """
    Calculate Mean Squared Error (MSE) between actual and predicted values.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
    
    Returns:
        float: The mean squared error
    """
    return np.mean((actual - predicted)**2)


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """
    Calculate Root Mean Squared Error (RMSE) between actual and predicted values.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
    
    Returns:
        float: The root mean squared error
    """
    return np.sqrt(mse(actual, predicted))


def mae(error: np.ndarray):
    """
    Calculate Mean Absolute Error (MAE) from an error array.
    
    Args:
        error: Array of error values
    
    Returns:
        float: The mean absolute error
    """
    return np.mean(np.abs(error))


class Rosbag2Reader:
    """
    A utility class to iterate over messages in a ROS 2 bag file.
    
    This class provides functionality to:
    - Open and read ROS 2 bag files
    - Filter messages by topic
    - Deserialize and return messages with their timestamps
    """

    def __init__(self, path, topics_filter=[], storage_id="sqlite3", serialization_format="cdr"):
        """
        Initialize the Rosbag2Reader.
        
        Args:
            path: Path to the ROS 2 bag directory
            topics_filter: List of topic names to filter (empty list = all topics)
            storage_id: Storage format identifier (default: "sqlite3")
            serialization_format: Message serialization format (default: "cdr")
        """
        self.__path = path
        self.__set_rosbag_options(storage_id, serialization_format)
        self.__reader = rosbag2_py.SequentialReader()
        self.__reader.open(self.__storage_options, self.__converter_options)

        # Build a mapping of topic names to message types
        topic_types = self.__reader.get_all_topics_and_types()
        self.__type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        # Apply topic filter if specified
        self.set_filter(topics_filter)

    def __iter__(self):
        """Reset the reader to the beginning for iteration."""
        self.__reset_bag_reader()
        return self

    def __next__(self):
        """
        Read and deserialize the next message from the bag.
        
        Returns:
            tuple: (topic_name, deserialized_message, timestamp)
        
        Raises:
            StopIteration: When no more messages are available
        """
        if self.__reader.has_next():
            (topic, data, t) = self.__reader.read_next()
            msg_type = get_message(self.__type_map[topic])
            msg = deserialize_message(data, msg_type)
            return (topic, msg, t)
        else:
            raise StopIteration

    @property
    def path(self):
        """Get the path to the ROS 2 bag file."""
        return self.__path

    @property
    def all_topics(self):
        """Get a dictionary of all topics and their message types."""
        return self.__type_map

    @property
    def selected_topics(self):
        """Get the currently selected topics (filtered or all if no filter applied)."""
        if self.__storage_filter is None:
            return self.all_topics
        else:
            return self.__selected_topics            

    def set_filter(self, topics):
        """
        Apply a topic filter to only read specific topics.
        
        Args:
            topics: List of topic names to filter
        
        Raises:
            KeyError: If any specified topic does not exist in the bag file
        """
        if topics:
            try:
                # Validate that all requested topics exist in the bag
                self.__selected_topics = {topic: self.__type_map[topic] for topic in topics}
            except KeyError as e:
                raise KeyError(f"Could not find topic {e} in the rosbag file")
            # Create storage filter for efficient reading
            self.__storage_filter = rosbag2_py.StorageFilter(topics=topics)
        else:
            self.__storage_filter = None

        # Reset the reader to apply the new filter
        self.__reset_bag_reader()

    def reset_filter(self):
        """Remove all topic filters and read all topics."""
        self.__storage_filter = None
        self.__reader.reset_filter()
        self.__reset_bag_reader()

    def __set_rosbag_options(self, storage_id, serialization_format):
        """
        Configure ROS 2 bag storage and serialization options.
        
        Args:
            storage_id: Storage backend identifier
            serialization_format: Message serialization format
        """
        self.__storage_options = rosbag2_py.StorageOptions(uri=self.__path, storage_id=storage_id)

        self.__converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format,
        )

    def __reset_bag_reader(self):
        """Reset the bag reader and reapply any active filters."""
        self.__reader.open(self.__storage_options, self.__converter_options)
        if self.__storage_filter is not None:
            self.__reader.set_filter(self.__storage_filter)


# --- UTILITY FUNCTIONS ---

def quaternion_to_yaw(q):
    """
    Convert a geometry_msgs/Quaternion to yaw angle in radians.
    
    This function extracts the yaw rotation (rotation around Z-axis) 
    from a quaternion representation.
    
    Args:
        q: A geometry_msgs/Quaternion object with fields: x, y, z, w
    
    Returns:
        float: Yaw angle in radians, in the range [-pi, pi]
    """
    # Formula to extract yaw from quaternion (rotation around Z-axis)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_ekf_metrics(bag_path, ekf_topic, truth_topic):
    """
    Compute RMSE and MAE metrics between EKF estimated pose and ground truth pose.
    
    This function reads a ROS 2 bag file, synchronizes EKF and ground truth poses,
    and computes error metrics for position (X, Y) and orientation (theta).
    
    Args:
        bag_path: Path to the ROS 2 bag directory
        ekf_topic: Topic name publishing EKF estimated pose (e.g., '/ekf/pose')
        truth_topic: Topic name publishing ground truth pose (e.g., '/ground_truth/pose')
    
    Note:
        Both topics should publish nav_msgs/Odometry or geometry_msgs/PoseStamped messages
    """
    try:
        # 1. Initialize the bag reader and filter topics
        reader = Rosbag2Reader(bag_path, topics_filter=[ekf_topic, truth_topic])
        print(f"Reading bag: {bag_path}")
        print(f"EKF Topic: {ekf_topic}, Truth Topic: {truth_topic}")

    except Exception as e:
        print(f"Error initializing Rosbag2Reader: {e}")
        return

    # Data structures to store poses indexed by their timestamp
    # Format: {timestamp: [x, y, theta]}
    ekf_data = {}    # EKF estimated poses
    truth_data = {}  # Ground truth poses

    # 2. Extract pose data from the bag file
    for topic, msg, t in reader:
        # Extract pose based on message structure
        # Messages can be nav_msgs/Odometry or geometry_msgs/PoseStamped
        if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
            # nav_msgs/Odometry: msg.pose.pose contains the Pose
            pose = msg.pose.pose
        elif hasattr(msg, 'pose'):
            # geometry_msgs/PoseStamped: msg.pose contains the Pose
            pose = msg.pose
        else:
            # Skip messages that don't have pose information
            continue

        # Extract position components (X, Y)
        x = pose.position.x
        y = pose.position.y
        
        # Extract orientation as yaw angle (rotation around Z-axis)
        theta = quaternion_to_yaw(pose.orientation)
        
        # Store the pose data indexed by message timestamp
        if topic == ekf_topic:
            ekf_data[t] = np.array([x, y, theta])
        elif topic == truth_topic:
            truth_data[t] = np.array([x, y, theta])

    # 3. Synchronize timestamps and extract corresponding poses
    
    # Use ground truth timestamps as the reference timeline
    truth_timestamps = sorted(truth_data.keys())
    
    # Arrays to store synchronized pose pairs
    synced_truth = []
    synced_ekf = []
    
    # Time tolerance for synchronization (5 milliseconds in nanoseconds)
    time_tolerance = 5 * 10**6 

    # For each ground truth measurement, find the closest EKF measurement
    for t_truth in truth_timestamps:
        # Find the EKF timestamp closest to this ground truth timestamp
        closest_t_ekf = min(ekf_data.keys(), key=lambda t_ekf: abs(t_ekf - t_truth))
        
        # Only include the pair if they are within the time tolerance
        if abs(t_truth - closest_t_ekf) <= time_tolerance:
            synced_truth.append(truth_data[t_truth])
            synced_ekf.append(ekf_data[closest_t_ekf])

    # Verify that synchronization was successful
    if not synced_truth:
        print("Error: Could not synchronize any data points between EKF and Ground Truth topics.")
        return

    # Convert lists of pose arrays to NumPy arrays for efficient computation
    # Shape: (num_samples, 3) where columns are [x, y, theta]
    synced_truth = np.array(synced_truth)
    synced_ekf = np.array(synced_ekf)
    
    # Calculate error matrix: Error = Ground_Truth - EKF_Estimate
    error_matrix = synced_truth - synced_ekf

    # Normalize angular error to the range (-pi, pi]
    # This prevents large errors when wrapping occurs (e.g., -pi to +pi)
    error_matrix[:, 2] = np.arctan2(np.sin(error_matrix[:, 2]), np.cos(error_matrix[:, 2]))
    
    # 4. Compute metrics and display results
    
    # Extract individual components for metric calculation
    truth_x, truth_y, truth_theta = synced_truth.T
    ekf_x, ekf_y, ekf_theta = synced_ekf.T
    error_x, error_y, error_theta = error_matrix.T
    
    # Compute combined position error (Euclidean distance between poses)
    pos_error = np.sqrt(error_x**2 + error_y**2)
    
    # Print results
    print("\n--- EKF Metric Results ---")
    print(f"Total synchronized samples: {len(synced_truth)}")
    print("--------------------------")
    
    # Position metrics (X and Y combined as Euclidean distance)
    print("POSITION METRICS (X and Y combined):")
    print(f"  RMSE (Position): {rmse(pos_error, np.zeros_like(pos_error)):.4f} m")
    print(f"  MAE (Position):  {mae(pos_error):.4f} m")
    
    # Individual component metrics
    print("\nINDIVIDUAL METRICS:")
    print(f"  RMSE (X):        {rmse(truth_x, ekf_x):.4f} m")
    print(f"  MAE (X):         {mae(error_x):.4f} m")
    
    print(f"  RMSE (Y):        {rmse(truth_y, ekf_y):.4f} m")
    print(f"  MAE (Y):         {mae(error_y):.4f} m")

    # Orientation metrics
    print(f"  RMSE (Theta):    {rmse(truth_theta, ekf_theta):.4f} rad")
    print(f"  MAE (Theta):     {mae(error_theta):.4f} rad")
    print("--------------------------")


def main():
    """
    Main entry point for EKF metrics computation.
    
    Configures the bag file path and topic names, then computes metrics.
    """
    # Path to the ROS 2 bag file containing EKF and ground truth data
    BAG_PATH = '/home/giorgia/Downloads/rosbag_task1_sim' 
    
    # Topic names for EKF output and ground truth
    EKF_TOPIC = '/ekf'           
    TRUTH_TOPIC = '/ground_truth'     

    # Compute and display metrics
    compute_ekf_metrics(BAG_PATH, EKF_TOPIC, TRUTH_TOPIC)


if __name__ == '__main__':
    main()
