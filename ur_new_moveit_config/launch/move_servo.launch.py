import os
import yaml

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch

def load_yaml(package_name: str, file_path: str):
    # Utility to load a YAML file from a ROS package
    pkg_path = get_package_share_directory(package_name)
    abs_path = os.path.join(pkg_path, file_path)
    try:
        with open(abs_path, 'r') as f:
            return yaml.safe_load(f)
    except OSError:
        return {}


def generate_launch_description():
    ld = LaunchDescription([
        DeclareLaunchArgument('launch_rviz', default_value='true', description='Launch RViz?'),
        DeclareLaunchArgument('launch_servo', default_value='false', description='Enable MoveIt Servo?'),
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation time?'),
    ])

    launch_servo = LaunchConfiguration('launch_servo')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Build MoveIt configuration
    builder = MoveItConfigsBuilder('ur_gripper', package_name='ur_new_moveit_config')
    builder.sensors_3d(file_path='config/sensors_3d.yaml')
    moveit_config = builder.to_moveit_configs()

    # Core MoveGroup launch
    move_group_launch = generate_move_group_launch(moveit_config)
    ld.add_action(move_group_launch)

    # Log servo flag
    ld.add_action(LogInfo(msg=["launch_servo set to: ", launch_servo]))

        # Servo node parameters: load YAML file into nested 'moveit_servo' namespace
    raw_servo = load_yaml('ur_new_moveit_config', 'config/ur_servo.yaml') or {}
    servo_params = {'moveit_servo': raw_servo}

    servo_node = Node(
        package='moveit_servo',
        executable='servo_node',
        name='moveit_servo_node',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            servo_params,
            {'use_sim_time': use_sim_time}
        ],
        condition=IfCondition(launch_servo),
    )
    ld.add_action(servo_node)

    return ld
