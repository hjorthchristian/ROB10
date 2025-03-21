#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def main():
    # Calibration results (Camera to Base)
    camera_to_base_rotation = np.array([
        [0.9966, 0.0816, 0.0124],
        [0.0818, -0.9965, -0.0162],
        [0.011, 0.0172, -0.9998]
    ])
    camera_to_base_translation = np.array([0.4951, 0.062, 1.1159])

    # Create 4x4 transformation matrix (Camera to Base)
    camera_to_base_transform = np.eye(4)
    camera_to_base_transform[:3, :3] = camera_to_base_rotation
    camera_to_base_transform[:3, 3] = camera_to_base_translation

    print("Camera to Base transformation matrix:")
    print(camera_to_base_transform)

    # Invert to get Base to Camera transformation
    base_to_camera_transform = np.linalg.inv(camera_to_base_transform)
    base_to_camera_rotation = base_to_camera_transform[:3, :3]
    base_to_camera_translation = base_to_camera_transform[:3, 3]

    print("\nBase to Camera transformation matrix:")
    print(base_to_camera_transform)
    print("\nBase to Camera translation (meters):")
    print(base_to_camera_translation)

    # Convert rotation matrix to different representations
    r = R.from_matrix(base_to_camera_rotation)

    # Different Euler angle representations
    euler_xyz = r.as_euler('xyz', degrees=True)  # Roll (X), Pitch (Y), Yaw (Z)
    euler_zyx = r.as_euler('zyx', degrees=True)  # Yaw (Z), Pitch (Y), Roll (X)
    euler_xyz_rad = r.as_euler('xyz', degrees=False)  # In radians for the command

    print("\nBase to Camera Euler Angles (degrees, XYZ order):")
    print(euler_xyz)
    print("\nBase to Camera Euler Angles (degrees, ZYX order):")
    print(euler_zyx)

    # Quaternion representation [x, y, z, w]
    quat = r.as_quat()
    print("\nBase to Camera Quaternion [x, y, z, w]:")
    print(quat)

    # Print various commands for static_transform_publisher
    print("\nCommand using Euler angles (XYZ, radians):")
    print(f"ros2 run tf2_ros static_transform_publisher {base_to_camera_translation[0]:.4f} {base_to_camera_translation[1]:.4f} {base_to_camera_translation[2]:.4f} {euler_xyz_rad[0]:.4f} {euler_xyz_rad[1]:.4f} {euler_xyz_rad[2]:.4f} ur10_base_link camera_link")

    print("\nCommand using quaternion (recommended for ROS2):")
    print(f"ros2 run tf2_ros static_transform_publisher {base_to_camera_translation[0]:.4f} {base_to_camera_translation[1]:.4f} {base_to_camera_translation[2]:.4f} {quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f} ur10_base_link camera_link")
    
    # Try different adjustments
    print("\n--- Common rotation adjustments to try ---")
    try_adjustment(base_to_camera_rotation, base_to_camera_translation, 180, 0, 0)  # Flip X axis
    try_adjustment(base_to_camera_rotation, base_to_camera_translation, 0, 180, 0)  # Flip Y axis
    try_adjustment(base_to_camera_rotation, base_to_camera_translation, 0, 0, 180)  # Flip Z axis
    try_adjustment(base_to_camera_rotation, base_to_camera_translation, 90, 0, 0)   # Rotate 90° around X
    try_adjustment(base_to_camera_rotation, base_to_camera_translation, 0, 90, 0)   # Rotate 90° around Y
    try_adjustment(base_to_camera_rotation, base_to_camera_translation, 0, 0, 90)   # Rotate 90° around Z
    try_adjustment(base_to_camera_rotation, base_to_camera_translation, 90, 0, 90)  # Rotate 90° around X and Z
    try_adjustment(base_to_camera_rotation, base_to_camera_translation, 180, 0, 90)  # Rotate 180° around X and 90° around Z

def try_adjustment(base_rotation, base_translation, rx_deg, ry_deg, rz_deg):
    """Apply additional rotation to the current transform and print command"""
    # Convert degrees to radians
    rx, ry, rz = map(math.radians, [rx_deg, ry_deg, rz_deg])
    
    # Create rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combine the rotations (post-multiply)
    adjustment = Rz @ Ry @ Rx
    
    # Apply to the current rotation
    new_rotation = base_rotation @ adjustment
    
    # Convert to quaternion
    r_new = R.from_matrix(new_rotation)
    quat_new = r_new.as_quat()
    
    print(f"\nAdjusted by rotating X:{rx_deg}°, Y:{ry_deg}°, Z:{rz_deg}°")
    print(f"ROS2 command with quaternion:")
    print(f"ros2 run tf2_ros static_transform_publisher {base_translation[0]:.4f} {base_translation[1]:.4f} {base_translation[2]:.4f} {quat_new[0]:.4f} {quat_new[1]:.4f} {quat_new[2]:.4f} {quat_new[3]:.4f} ur10_base_link camera_link")

if __name__ == "__main__":
    main()