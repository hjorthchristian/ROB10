from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Create builder
    builder = MoveItConfigsBuilder("ur_gripper", package_name="ur_new_moveit_config")
    
    # Configure builder
    # builder.sensors_3d(file_path="config/sensors_3d.yaml")
    
    # Convert to MoveItConfigs
    moveit_config = builder.to_moveit_configs()
    
    # Modify the configuration dictionaries
    moveit_config.move_group_capabilities["capabilities"] = "move_group/ExecuteTaskSolutionCapability"
    # moveit_config.planning_scene_monitor["octomap_frame"] = "world"
    # moveit_config.planning_scene_monitor["octomap_resolution"] = 0.05
    # moveit_config.planning_scene_monitor["max_range"] = 5.0

    return generate_move_group_launch(moveit_config)