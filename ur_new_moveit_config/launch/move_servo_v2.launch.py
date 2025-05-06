from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path) as file:
            return yaml.safe_load(file)
    except OSError:
        return None

def generate_launch_description():
    # Declare arguments
    declared_arguments = [
        DeclareLaunchArgument(
            "launch_servo", default_value="true", description="Launch Servo?"
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Using or not time from simulation",
        ),
    ]
    
    # Get launch configuration
    launch_servo = LaunchConfiguration("launch_servo")
    use_sim_time = LaunchConfiguration("use_sim_time")
    
    # Create builder
    builder = MoveItConfigsBuilder("ur_gripper", package_name="ur_new_moveit_config")
    
    # Configure builder
    builder.sensors_3d(file_path="config/sensors_3d.yaml")
    
    # Convert to MoveItConfigs
    moveit_config = builder.to_moveit_configs()
    
    # Modify the configuration dictionaries
    moveit_config.move_group_capabilities["capabilities"] = "move_group/ExecuteTaskSolutionCapability"
    moveit_config.planning_scene_monitor["octomap_frame"] = "world"
    moveit_config.planning_scene_monitor["octomap_resolution"] = 0.05
    moveit_config.planning_scene_monitor["max_range"] = 5.0
    
    # Load servo configuration yaml
    servo_yaml = load_yaml("ur_new_moveit_config", "config/ur_servo.yaml")
    if not servo_yaml:
        # If the config doesn't exist in the package, create default configuration
        servo_yaml = {
            "cartesian_command_in_topic": "servo_server/delta_twist_cmds",
            "joint_command_in_topic": "servo_server/delta_joint_cmds",
            "command_out_topic": "servo_server/command",
            "publish_period": 0.01,
            "planning_frame": "ur10_base_link",
            "robot_link_command_frame": "ur10_tcp",
            "check_collisions": True,
        }
    
    # Create servo node
    servo_params = {"moveit_servo": servo_yaml}
    servo_node = Node(
        package="moveit_servo",
        condition=IfCondition(launch_servo),
        executable="servo_node",
        parameters=[
            moveit_config.to_dict(),
            servo_params,
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )
    
    # Create launch description with move group and servo components
    ld = LaunchDescription(declared_arguments)
    
    # Add servo node
    ld.add_action(servo_node)
    
    # Add the move_group launch components
    move_group_launch = generate_move_group_launch(moveit_config)
    
    # Combine the move_group launch with our launch description
    for action in move_group_launch._actions:
        ld.add_action(action)
    
    return ld
