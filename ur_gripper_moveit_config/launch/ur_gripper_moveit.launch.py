import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Start the actual robot control stack (includes ur_robot_driver)
    robot_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_gripper_control'), 'launch', 'start_robot.launch.py'
            ])
        ])
    )

    # Start the move_group node 
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_gripper_moveit_config'), 'launch', 'move_group.launch.py'
            ])
        ])
    )
    
    # Start RViz
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_gripper_moveit_config'), 'launch', 'moveit_rviz.launch.py'
            ])
        ])
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        robot_driver_launch,
        move_group_launch,
        rviz_launch
    ])