from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'),
            '/launch/rs_launch.py'
        ])
    )

    rviz_config_file = os.path.join(
        FindPackageShare('moveit_ur').find('moveit_ur'),
        'config',
        'realsense_default.rviz'
    )

    rviz_node = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config_file],
        output='screen'
    )

    return LaunchDescription([
        realsense_launch,
        rviz_node
    ])
