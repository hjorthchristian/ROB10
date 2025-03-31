from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("ur_gripper", package_name="ur_new_moveit_config").to_moveit_configs()
    
    # Add the MTC capability
    moveit_config.move_group_capabilities["capabilities"] = ["move_group/ExecuteTaskSolutionCapability"]
    
    return generate_move_group_launch(moveit_config)
