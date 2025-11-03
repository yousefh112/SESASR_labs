import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    """
    Generate a launch description that starts the TurtleBot3 simulation
    and the bump-and-go controller node.
    """

    # --- 1. Configure the Bump-and-Go Controller Node ---

    # Retrieve the package share directory for lab02_pkg
    my_pkg_name = 'lab02_pkg'
    my_pkg_share = get_package_share_directory(my_pkg_name)

    # Construct the full path to the bump-and-go controller parameters file
    params_file_path = os.path.join(
        my_pkg_share,
        'config',
        'bump_and_go_params.yaml'
    )

    # Create the bump-and-go node action with parameters from the YAML config
    start_bump_and_go_node = Node(
        package=my_pkg_name,
        executable='bump_and_go_node',
        name='bump_and_go_node',
        output='screen',
        parameters=[params_file_path]
    )

    # --- 2. Configure the Gazebo Simulation Launch File ---

    # Retrieve the package share directory for the TurtleBot3 Gazebo simulation
    sim_pkg_name = 'turtlebot3_gazebo'
    sim_pkg_share = get_package_share_directory(sim_pkg_name)

    # Construct the full path to the simulation's launch file
    sim_launch_file_path = os.path.join(
        sim_pkg_share,
        'launch',
        'lab02.launch.py'
    )

    # Define the include launch description action
    start_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_file_path)
    )

    # --- 3. Create the LaunchDescription ---

    # Return the LaunchDescription, which will start both actions
    return LaunchDescription([
        start_simulation,         # Start the simulation first
        start_bump_and_go_node    # Then start your controller
    ])