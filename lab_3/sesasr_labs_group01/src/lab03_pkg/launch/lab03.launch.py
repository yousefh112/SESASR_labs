import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    """
    Generate a launch description that starts the TurtleBot3 in real life mode
    and the bump-and-go controller node.
    """

    # Retrieve the package share directory for lab03_pkg
    my_pkg_name = 'lab03_pkg'
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
        executable='bump_go_real',
        name='bump_go_real',
        output='screen',
        parameters=[params_file_path]
    )

    #


    # Return the LaunchDescription, which will start both actions
    return LaunchDescription([
        start_bump_and_go_node    # Then start your controller
    ])