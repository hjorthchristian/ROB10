from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_setup_assistant_launch
from launch import LaunchDescription
from launch.actions import LogInfo, ExecuteProcess
import os
import xml.etree.ElementTree as ET
import yaml


def generate_launch_description():
    launch_elements = [
        LogInfo(msg="Inspecting MoveIt configuration for ur_gripper")
    ]
    
    # Find URDF and SRDF files
    try:
        package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(package_path, "config")
        
        # Check SRDF
        srdf_path = os.path.join(config_dir, "ur_gripper.srdf")
        if os.path.exists(srdf_path):
            launch_elements.append(LogInfo(msg=f"Found SRDF: {srdf_path}"))
            
            # Parse SRDF
            try:
                tree = ET.parse(srdf_path)
                root = tree.getroot()
                
                # Check for group definitions
                groups = root.findall(".//group")
                launch_elements.append(LogInfo(msg=f"Found {len(groups)} groups in SRDF"))
                
                # Check for joint definitions in groups
                for group in groups:
                    group_name = group.get('name', 'unnamed')
                    joint_tags = group.findall(".//joint")
                    launch_elements.append(LogInfo(msg=f"Group '{group_name}' has {len(joint_tags)} joint tags"))
                    
                    # Check each joint
                    for joint in joint_tags:
                        joint_name = joint.get('name', 'unnamed')
                        launch_elements.append(LogInfo(msg=f"  Joint: {joint_name}"))
                        
                        # Check if joint has a name
                        if not joint.get('name'):
                            launch_elements.append(LogInfo(msg=f"  WARNING: Joint without name in group {group_name}"))
            except Exception as srdf_err:
                launch_elements.append(LogInfo(msg=f"Error parsing SRDF: {str(srdf_err)}"))
        else:
            launch_elements.append(LogInfo(msg="SRDF file not found"))
        
        # Check URDF
        urdf_path = os.path.join(config_dir, "ur_gripper.urdf.xacro")
        if os.path.exists(urdf_path):
            launch_elements.append(LogInfo(msg=f"Found URDF: {urdf_path}"))
            
            # Run xacro to expand the URDF
            launch_elements.append(LogInfo(msg="Running xacro to expand URDF"))
            xacro_process = ExecuteProcess(
                cmd=['xacro', urdf_path],
                output='screen',
                shell=False
            )
            launch_elements.append(xacro_process)
        else:
            launch_elements.append(LogInfo(msg="URDF file not found"))
        
        # Check kinematics.yaml
        kinematics_path = os.path.join(config_dir, "kinematics.yaml")
        if os.path.exists(kinematics_path):
            launch_elements.append(LogInfo(msg=f"Found kinematics config: {kinematics_path}"))
            
            # Parse YAML
            try:
                with open(kinematics_path, 'r') as file:
                    kinematics_data = yaml.safe_load(file)
                    launch_elements.append(LogInfo(msg=f"Kinematics config contents: {kinematics_data}"))
            except Exception as yaml_err:
                launch_elements.append(LogInfo(msg=f"Error parsing kinematics YAML: {str(yaml_err)}"))
        else:
            launch_elements.append(LogInfo(msg="Kinematics config not found"))
        
    except Exception as e:
        launch_elements.append(LogInfo(msg=f"Error inspecting files: {str(e)}"))
    
    # Create the MoveIt configuration
    try:
        moveit_config = MoveItConfigsBuilder("ur_gripper", package_name="ur10_gripper_moveit_config").to_moveit_configs()
        setup_assistant_launch = generate_setup_assistant_launch(moveit_config)
        
        for element in setup_assistant_launch.entities:
            launch_elements.append(element)
            
    except Exception as e:
        launch_elements.append(LogInfo(msg=f"Error generating setup assistant launch: {str(e)}"))
    
    return LaunchDescription(launch_elements)