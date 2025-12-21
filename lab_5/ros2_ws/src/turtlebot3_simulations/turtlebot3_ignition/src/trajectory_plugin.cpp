#include <cmath>
#include <chrono>
#include <thread>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <ignition/gazebo/EntityComponentManager.hh>
#include <ignition/gazebo/Link.hh>
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/plugin/Register.hh>
#include <sdf/Element.hh>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>

namespace trajectory_plugin
{
/**
 * @brief Cubic Hermite spline interpolation for 3D trajectory
 * Uses time-parameterized waypoints to compute smooth position along trajectory
 */
class TimedSpline
{
public:
  struct Point 
  { 
    double t;              // Time parameter
    Eigen::Vector3d p;     // 3D position
  };

  explicit TimedSpline(const std::vector<Point>& pts)
  : points_(pts)
  {
    if (pts.size() < 2) {
      throw std::runtime_error("TimedSpline requires at least 2 points");
    }
    computeTangents();
  }

  /**
   * @brief Compute interpolated position at given time
   * @param t Time parameter (wraps around trajectory period)
   * @return Interpolated 3D position
   */
  Eigen::Vector3d position(double t) const
  {
    t = wrapTime(t);

    int i = segmentIndex(t);
    int j = i + 1;

    double t0 = points_[i].t;
    double t1 = points_[j].t;
    double dt = t1 - t0;

    // Normalize time to [0, 1] within segment
    double s = (t - t0) / dt;

    const auto& p0 = points_[i].p;
    const auto& p1 = points_[j].p;
    const auto& m0 = tangents_[i];
    const auto& m1 = tangents_[j];

    // Cubic Hermite basis functions
    double s2 = s * s;
    double s3 = s2 * s;
    double h00 = 2 * s3 - 3 * s2 + 1;      // Position at start
    double h10 = s3 - 2 * s2 + s;          // Tangent at start
    double h01 = -2 * s3 + 3 * s2;         // Position at end
    double h11 = s3 - s2;                  // Tangent at end

    return h00 * p0 + h10 * dt * m0 + h01 * p1 + h11 * dt * m1;
  }

private:
  std::vector<Point> points_;
  std::vector<Eigen::Vector3d> tangents_;

  /**
   * @brief Compute tangents at each waypoint using finite differences
   */
  void computeTangents()
  {
    const size_t N = points_.size();
    tangents_.resize(N);

    for (size_t i = 0; i < N; ++i) {
      if (i == 0) {
        // Forward difference at start
        tangents_[i] = (points_[1].p - points_[0].p) / (points_[1].t - points_[0].t);
      } else if (i == N - 1) {
        // Backward difference at end
        tangents_[i] = (points_[N - 1].p - points_[N - 2].p) / (points_[N - 1].t - points_[N - 2].t);
      } else {
        // Central difference in the middle
        tangents_[i] = (points_[i + 1].p - points_[i - 1].p) / (points_[i + 1].t - points_[i - 1].t);
      }
    }
  }

  /**
   * @brief Wrap time to trajectory period [0, T]
   */
  double wrapTime(double t) const
  {
    const double T = points_.back().t;
    t = fmod(t, T);
    if (t < 0) { t += T; }
    return t;
  }

  /**
   * @brief Find the segment index for given time
   */
  int segmentIndex(double t) const
  {
    for (size_t i = 0; i < points_.size() - 1; ++i) {
      if (t >= points_[i].t && t <= points_[i + 1].t) {
        return static_cast<int>(i);
      }
    }
    return static_cast<int>(points_.size() - 2);
  }
};

/**
 * @brief Gazebo plugin for moving objects along predefined trajectories
 * Publishes pose to ROS2 topic for external tracking
 */
class TrajectoryPlugin : public ignition::gazebo::System,
  public ignition::gazebo::ISystemConfigure,
  public ignition::gazebo::ISystemPreUpdate
{
public:
  TrajectoryPlugin() = default;
  
  ~TrajectoryPlugin()
  {
    // Shutdown ROS
    if (node_) {
      rclcpp::shutdown();
    }
    if (ros_spinner_.joinable()) {
      ros_spinner_.join();
    }
  }

  void Configure(
    const ignition::gazebo::Entity& entity,
    const std::shared_ptr<const sdf::Element>& sdf,
    ignition::gazebo::EntityComponentManager& ecm,
    ignition::gazebo::EventManager& /*eventMgr*/) override
  {
    // Initialize model and link
    model_ = ignition::gazebo::Model(entity);
    if (!model_.Valid(ecm)) {
      throw std::runtime_error("Invalid model entity");
    }

    // Parse SDF configuration
    if (!sdf->HasElement("script")) {
      throw std::runtime_error("Missing <script> element in plugin SDF");
    }
    auto scriptElem = const_cast<sdf::Element*>(sdf.get())->GetElement("script");

    loop_ = scriptElem->Get<bool>("loop", true).first;
    auto_start_ = scriptElem->Get<bool>("auto_start", true).first;

    // Parse trajectory waypoints
    if (!scriptElem->HasElement("trajectory")) {
      throw std::runtime_error("Missing <trajectory> element in <script>");
    }
    auto trajElem = scriptElem->GetElement("trajectory");

    std::vector<TimedSpline::Point> waypoints;
    auto waypointElem = trajElem->GetElement("waypoint");
    
    while (waypointElem) {
      const double time = waypointElem->Get<double>("time");
      const std::string poseStr = waypointElem->Get<std::string>("pose");
      
      // Parse pose: x y z roll pitch yaw
      std::stringstream ss(poseStr);
      double x, y, z, roll, pitch, yaw;
      if (!(ss >> x >> y >> z >> roll >> pitch >> yaw)) {
        throw std::runtime_error("Invalid pose format in waypoint");
      }

      TimedSpline::Point point;
      point.t = time;
      point.p = Eigen::Vector3d(x, y, z);
      waypoints.push_back(point);

      waypointElem = waypointElem->GetNextElement("waypoint");
    }

    if (waypoints.size() < 2) {
      throw std::runtime_error("Trajectory requires at least 2 waypoints");
    }
    spline_ = std::make_unique<TimedSpline>(waypoints);

    // Initialize ROS2 node and publisher
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
    
    node_ = std::make_shared<rclcpp::Node>("trajectory_publisher");
    
    // Enable simulation time for synchronization with Gazebo
    node_->set_parameter(rclcpp::Parameter("use_sim_time", true));
    
    pose_publisher_ = node_->create_publisher<nav_msgs::msg::Odometry>(
      "/dynamic_goal_pose", 10);
    
    // Create timer for publishing pose at 10 Hz
    const auto publish_period = std::chrono::milliseconds(100);
    pose_timer_ = node_->create_wall_timer(
      publish_period,
      [this]() { publishCurrentPose(); });

    // Start ROS2 spin in separate thread
    ros_spinner_ = std::thread([this]() { 
      rclcpp::spin(node_); 
    });

    start_time_ = std::chrono::steady_clock::duration::zero();
  }

  void PreUpdate(
    const ignition::gazebo::UpdateInfo& info,
    ignition::gazebo::EntityComponentManager& ecm) override
  {
    // Initialize start time on first update
    if (start_time_ == std::chrono::steady_clock::duration::zero()) {
      start_time_ = info.simTime;
    }

    // Compute elapsed time in seconds
    const double elapsed_time = std::chrono::duration<double>(info.simTime - start_time_).count();

    // Interpolate position along trajectory
    const Eigen::Vector3d position = spline_->position(elapsed_time);
    const ignition::math::Pose3d pose(position.x(), position.y(), position.z(), 0, 0, 0);

    // Update model pose in simulation
    model_.SetWorldPoseCmd(ecm, pose);

    // Store current position for ROS publishing
    current_position_.store(position);
  }

private:
  /**
   * @brief Publish current pose to ROS2 topic
   */
  void publishCurrentPose()
  {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = node_->now();
    msg.header.frame_id = "world";
    
    const Eigen::Vector3d pos = current_position_.load();
    msg.pose.pose.position.x = pos.x();
    msg.pose.pose.position.y = pos.y();
    msg.pose.pose.position.z = pos.z();
    msg.pose.pose.orientation.w = 1.0;  // Identity quaternion
    pose_publisher_->publish(msg);
  }

  // Gazebo entities
  ignition::gazebo::Model model_;

  // ROS2 communication
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_publisher_;
  rclcpp::TimerBase::SharedPtr pose_timer_;
  std::thread ros_spinner_;

  // Trajectory parameters
  std::unique_ptr<TimedSpline> spline_;
  std::chrono::steady_clock::duration start_time_;
  bool loop_{true};
  bool auto_start_{true};

  // Shared state between simulation and ROS threads
  struct AtomicVector3d {
    std::atomic<double> x{0.0};
    std::atomic<double> y{0.0};
    std::atomic<double> z{0.0};
    
    void store(const Eigen::Vector3d& v) {
      x.store(v.x());
      y.store(v.y());
      z.store(v.z());
    }
    
    Eigen::Vector3d load() const {
      return Eigen::Vector3d(x.load(), y.load(), z.load());
    }
  };
  AtomicVector3d current_position_;
};
}  // namespace trajectory_plugin

IGNITION_ADD_PLUGIN(
  trajectory_plugin::TrajectoryPlugin,
  ignition::gazebo::System,
  ignition::gazebo::ISystemConfigure,
  ignition::gazebo::ISystemPreUpdate
);
IGNITION_ADD_PLUGIN_ALIAS(trajectory_plugin::TrajectoryPlugin, "trajectory_plugin");