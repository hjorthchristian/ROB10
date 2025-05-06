from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # Load MoveIt configs
    moveit_config = MoveItConfigsBuilder("ur_gripper", package_name="ur_new_moveit_config").to_dict()


    # MTC Demo node
    mtc_demo = Node(
        package="ur10_mtc_demo",
        executable="pick_and_place_mtc_node",
        output="screen",
        parameters=[
            moveit_config,
        ],
    )

    return LaunchDescription([
        mtc_demo
    ])
