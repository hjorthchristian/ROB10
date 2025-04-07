from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # Load the MoveIt configuration (update the robot and package name if needed)
    moveit_config = MoveItConfigsBuilder("ur_gripper", package_name="ur_new_moveit_config").to_dict()

    get_tcp_pose_node = Node(
        package="moveit_ur",
        executable="move_to_pose_movegroup",
        output="screen",
        parameters=[moveit_config]
    )

    return LaunchDescription([
        get_tcp_pose_node
    ])