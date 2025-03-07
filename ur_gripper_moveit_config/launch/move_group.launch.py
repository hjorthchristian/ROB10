from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch import LaunchDescription
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("ur_gripper", package_name="ur_gripper_moveit_config").to_moveit_configs()
    
    # Load controller parameters
    controller_config = os.path.join(
        get_package_share_directory("ur_gripper_moveit_config"),
        "config",
        "controllers.yaml",
    )
    
    # Add controller parameters to moveit_configs
    if not hasattr(moveit_config, "trajectory_execution"):
        # Create the dictionary if it doesn't exist
        moveit_config.trajectory_execution = {}
    
    moveit_config.trajectory_execution["moveit_manage_controllers"] = True
    moveit_config.trajectory_execution["controller_list_name"] = "controllers.yaml"
    moveit_config.trajectory_execution["controller_list_file"] = controller_config
    
    return generate_move_group_launch(moveit_config)
