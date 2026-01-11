import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import numpy as np
import math
import os

# ==========================================
# 1. Utility Functions
# ==========================================

def mse(actual: np.ndarray, predicted: np.ndarray):
    """Calculate Mean Squared Error between actual and predicted values."""
    return np.mean((actual - predicted)**2)

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """Calculate Root Mean Squared Error between actual and predicted values."""
    return np.sqrt(mse(actual, predicted))

def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(angle):
    """Normalize angle to range [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

class Rosbag2Reader:
    """Reader class for ROS2 bag files with topic filtering support."""
    
    def __init__(self, path, topics_filter=[], storage_id="sqlite3", serialization_format="cdr"):
        """
        Initialize the rosbag2 reader.
        
        Args:
            path: Path to the rosbag2 file
            topics_filter: List of topic names to filter (empty list means all topics)
            storage_id: Storage format identifier (default: sqlite3)
            serialization_format: Message serialization format (default: cdr)
        """
        self.__path = path
        self.__storage_options = rosbag2_py.StorageOptions(uri=self.__path, storage_id=storage_id)
        self.__converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format,
        )
        self.__reader = rosbag2_py.SequentialReader()
        self.__reader.open(self.__storage_options, self.__converter_options)

        # Build a map of topic names to their message types
        topic_types = self.__reader.get_all_topics_and_types()
        self.__type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
        
        # Apply topic filter if provided
        if topics_filter:
            valid_topics = [t for t in topics_filter if t in self.__type_map]
            self.__storage_filter = rosbag2_py.StorageFilter(topics=valid_topics)
            self.__reader.set_filter(self.__storage_filter)

    def __iter__(self):
        """Make the reader iterable."""
        return self

    def __next__(self):
        """
        Read next message from the bag file.
        
        Returns:
            Tuple of (topic_name, deserialized_message, timestamp)
        """
        if self.__reader.has_next():
            (topic, data, t) = self.__reader.read_next()
            if topic not in self.__type_map:
                return self.__next__()
            
            # Deserialize the message
            msg_type_name = self.__type_map[topic]
            try:
                msg_type = get_message(msg_type_name)
                msg = deserialize_message(data, msg_type)
            except Exception:
                return self.__next__()
                
            return (topic, msg, t)
        else:
            raise StopIteration

# ==========================================
# 2. DWA Analysis Node 
# ==========================================

class DwaAnalysisNode(Node):
    """
    ROS2 Node for analyzing Dynamic Window Approach (DWA) robot navigation performance.
    
    Analyzes rosbag2 files to compute tracking metrics and obstacle safety metrics.
    """
    
    def __init__(self):
        super().__init__('dwa_analysis_node')

        # ---------------------------------------------------------
        # Rosbag path configuration - MODIFY PATH HERE
        # ---------------------------------------------------------
        default_bag_path = "/bags/rosbag2_2026_01_08-17_22_58"
        
        # Declare ROS2 parameters with defaults
        self.declare_parameter('bag_path', default_bag_path)
        self.declare_parameter('is_simulation', False)  # True: simulation, False: real robot
        self.declare_parameter('task_type', 'task3')   # 'task1' or 'task2'

        # Get parameter values
        self.bag_path = self.get_parameter('bag_path').value
        self.is_simulation = self.get_parameter('is_simulation').value
        self.task_type = self.get_parameter('task_type').value

        # Log initialization info
        self.get_logger().info(f"Initialized Analysis Node.")
        self.get_logger().info(f"Mode: {'SIMULATION' if self.is_simulation else 'REAL ROBOT'}")
        self.get_logger().info(f"Analyzing Bag: {self.bag_path}")

        # Start analysis
        self.analyze_dwa_task()

    def analyze_dwa_task(self):
        """Main analysis method that extracts data from rosbag and computes metrics."""
        
        # Define topic names
        SCAN_TOPIC = '/scan' 
        
        # Topic names depend on whether using simulation or real robot
        if self.is_simulation:
            ROBOT_TOPIC = '/ground_truth'
            TARGET_TOPIC = '/dynamic_goal_pose' 
        else:
            ROBOT_TOPIC = '/odom'
            TARGET_TOPIC = '/camera/landmarks'

        # Verify rosbag file exists
        if not os.path.exists(self.bag_path):
            self.get_logger().error(f"Rosbag path does not exist: {self.bag_path}")
            return

        # Open rosbag file with topic filtering
        try:
            reader = Rosbag2Reader(self.bag_path, topics_filter=[ROBOT_TOPIC, TARGET_TOPIC, SCAN_TOPIC])
        except Exception as e:
            self.get_logger().error(f"Failed to open bag: {e}")
            return

        # Initialize data storage structures
        robot_data = {}        # {timestamp: [x, y, yaw]}
        target_data = {}       # {timestamp: [x, y] or [dist, bearing]}
        scan_min_history = []  # Minimum distance to obstacles at each scan
        
        self.get_logger().info("Extracting data from bag...")
        
        # Parse messages from rosbag
        for topic, msg, t in reader:
            # ===== A. Extract Robot Pose =====
            if topic == ROBOT_TOPIC:
                # Check if message has pose structure (Odometry or PoseWithCovariance)
                if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                    p = msg.pose.pose
                    x, y = p.position.x, p.position.y
                    yaw = quaternion_to_yaw(p.orientation)
                    robot_data[t] = np.array([x, y, yaw])
            
            # ===== B. Extract Target Pose =====
            elif topic == TARGET_TOPIC:
                if self.is_simulation:
                    # Simulation: extract [x, y] from PoseStamped
                    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                        p = msg.pose.pose
                        target_data[t] = np.array([p.position.x, p.position.y])
                else:
                    # Real robot: extract [distance, bearing] from landmarks
                    if hasattr(msg, 'landmarks') and len(msg.landmarks) > 0:
                        lm = msg.landmarks[0]
                        dist = lm.range
                        bearing = lm.bearing
                        target_data[t] = np.array([dist, bearing])
                    else:
                        target_data[t] = None
            
            # ===== C. Extract LaserScan Data =====
            elif topic == SCAN_TOPIC:
                if hasattr(msg, 'ranges'):
                    # Convert ranges to numpy array for processing
                    ranges = np.array(msg.ranges, dtype=float)
                    
                    # Step 1: Remove invalid values (Inf, NaN)
                    valid_mask = np.isfinite(ranges)
                    valid_ranges = ranges[valid_mask]
                    
                    if len(valid_ranges) > 0:
                        # Step 2: Filter out distant readings (max 3.5m)
                        valid_ranges = valid_ranges[valid_ranges <= 3.5]
                        
                        # Step 3: Filter out zero/very small readings
                        valid_ranges = valid_ranges[valid_ranges > 0.01]
                        
                        if len(valid_ranges) > 0:
                            # Step 4: Record minimum distance to obstacles
                            current_min = np.min(valid_ranges)
                            scan_min_history.append(current_min)

        # Check if sufficient data was extracted
        if not robot_data or not target_data:
            self.get_logger().warn("Missing Pose/Target data in bag.")
            return

        # ===== 4. Compute Tracking and Safety Metrics =====
        self.get_logger().info("Calculating Metrics...")
        
        # Sort timestamps and prepare for synchronization
        target_timestamps = sorted(target_data.keys())
        robot_timestamps = np.array(sorted(robot_data.keys()))
        
        # Storage for synchronized metrics
        synced_dist = []       # Distance from robot to target
        synced_bearing = []    # Bearing angle to target
        tracking_intervals = [] # Time intervals between tracking measurements
        
        # Configuration parameters
        OPTIMAL_DIST = 0.7 if self.task_type == 'task2' else 0.4  # Target distance
        OPTIMAL_BEARING = 0.0  # Target bearing angle
        SIM_SENSOR_LIMIT = 3.5  # Maximum sensor range for simulation
        prev_time = None

        # Process each target measurement
        for t_targ in target_timestamps:
            # Calculate time interval since previous measurement
            dt = (t_targ - prev_time) * 1e-9 if prev_time else 0.0
            prev_time = t_targ

            if self.is_simulation:
                # Simulation: match with closest robot pose measurement
                if len(robot_timestamps) == 0: 
                    continue
                    
                # Find closest robot timestamp
                idx = (np.abs(robot_timestamps - t_targ)).argmin()
                
                # Skip if timestamps are too far apart (> 0.1s)
                if abs(robot_timestamps[idx] - t_targ) > 0.1 * 1e9: 
                    continue 
                
                # Extract robot and target poses
                t_rob = robot_timestamps[idx]
                rx, ry, ryaw = robot_data[t_rob]
                tx, ty = target_data[t_targ]
                
                # Calculate distance to target
                curr_dist = math.sqrt((tx - rx)**2 + (ty - ry)**2)
                
                # Calculate bearing angle to target
                global_angle = math.atan2(ty - ry, tx - rx)
                curr_bearing = normalize_angle(global_angle - ryaw)
                
                synced_dist.append(curr_dist)
                synced_bearing.append(curr_bearing)
                
                # Only count intervals when target is within sensor range
                if curr_dist < SIM_SENSOR_LIMIT:
                    tracking_intervals.append(dt)
            else:
                # Real robot: use distance and bearing directly from sensors
                val = target_data[t_targ]
                if val is not None:
                    synced_dist.append(val[0])
                    synced_bearing.append(val[1])
                    tracking_intervals.append(dt)

        # ===== 5. Calculate Performance Metrics =====
        synced_dist = np.array(synced_dist)
        synced_bearing = np.array(synced_bearing)
        
        # Compute tracking error metrics
        if len(synced_dist) > 0:
            rmse_dist_val = rmse(synced_dist, np.full_like(synced_dist, OPTIMAL_DIST))
            avg_dist = np.mean(synced_dist)
            rmse_bearing_val = rmse(synced_bearing, np.full_like(synced_bearing, OPTIMAL_BEARING))
        else:
            rmse_dist_val = 0.0
            rmse_bearing_val = 0.0
            avg_dist = 0.0
            
        # Compute tracking percentage
        total_duration = (target_timestamps[-1] - target_timestamps[0]) * 1e-9 if len(target_timestamps) > 1 else 0
        tracked_duration = sum(tracking_intervals)
        tracking_pct = (tracked_duration / total_duration * 100) if total_duration > 0 else 0.0

        # Compute obstacle safety metrics
        scan_min_history = np.array(scan_min_history)
        if len(scan_min_history) > 0:
            # Minimum distance achieved (safety threshold)
            obst_min_dist = np.min(scan_min_history)
            # Average distance to obstacles
            obst_avg_dist = np.mean(scan_min_history)
        else:
            obst_min_dist = -1.0
            obst_avg_dist = -1.0

        # ===== 6. Generate and Print Report =====
        self.print_report(
            total_duration, len(synced_dist), 
            OPTIMAL_DIST, avg_dist, rmse_dist_val, rmse_bearing_val, tracking_pct,
            obst_min_dist, obst_avg_dist, len(scan_min_history)
        )

    def print_report(self, duration, samples, opt_dist, avg_dist, rmse_dist, rmse_bearing, track_pct, obst_min, obst_avg, scan_samples):
        """Print formatted analysis report to ROS2 logger."""
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info(f" REPORT | Mode: {'SIMULATION' if self.is_simulation else 'REAL ROBOT'}")
        self.get_logger().info("="*50)
        self.get_logger().info(f"Total Duration:   {duration:.2f} s")
        self.get_logger().info("-" * 50)
        
        self.get_logger().info("1. Tracking Metrics:")
        self.get_logger().info(f"   RMSE Dist:     {rmse_dist:.4f} m (Opt: {opt_dist}m)")
        self.get_logger().info(f"   RMSE Bearing:  {rmse_bearing:.4f} rad")
        self.get_logger().info(f"   Time of Track: {track_pct:.2f} %")
        
        self.get_logger().info("-" * 50)
        self.get_logger().info("2. Obstacle Metrics (/scan):")
        if obst_min != -1.0:
            self.get_logger().info(f"   Min Dist:      {obst_min:.4f} m  <-- [Safety Check]")
            self.get_logger().info(f"   Avg Dist:      {obst_avg:.4f} m")
            self.get_logger().info(f"   Scans used:    {scan_samples}")
        else:
            self.get_logger().info("   No valid scan data found.")
            
        self.get_logger().info("="*50)


def main(args=None):
    """Entry point for the ROS2 node."""
    rclpy.init(args=args)
    node = DwaAnalysisNode()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()