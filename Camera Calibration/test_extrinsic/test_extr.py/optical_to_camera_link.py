import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix."""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return r.as_matrix()

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    r = R.from_quat([qx, qy, qz, qw])
    return r.as_matrix()

def rotation_matrix_to_quaternion(matrix):
    """Convert rotation matrix to quaternion."""
    r = R.from_matrix(matrix)
    return r.as_quat()  # Returns x, y, z, w

def rotation_matrix_to_euler(matrix):
    """Convert rotation matrix to Euler angles."""
    r = R.from_matrix(matrix)
    return r.as_euler('xyz')

# Original transform from ur10_base_link to camera_link
translation = np.array([0.4402, 0.0595, 1.1127])
original_euler = np.array([-3.0580, 1.5628, -1.3798])

# Create homogeneous transformation matrix from original transform
original_rotation_matrix = euler_to_rotation_matrix(*original_euler)
original_transform = np.eye(4)
original_transform[:3, :3] = original_rotation_matrix
original_transform[:3, 3] = translation

print("Original transformation matrix:")
print(original_transform)

# Now let's add a local rotation around the x-axis
def apply_local_x_rotation(transform_matrix, angle_rad):
    """Apply a local rotation around the x-axis."""
    # Create rotation matrix for x-axis rotation
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    x_rotation = np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])
    
    # Extract the current rotation matrix
    current_rotation = transform_matrix[:3, :3]
    
    # Apply the local rotation (multiply on the right for local rotation)
    new_rotation = current_rotation @ x_rotation
    
    # Create the new transformation matrix
    new_transform = transform_matrix.copy()
    new_transform[:3, :3] = new_rotation
    
    return new_transform

# Example: rotate 30 degrees (π/6 radians) around local x-axis
local_rotation_angle = np.pi  # 30 degrees
new_transform = apply_local_x_rotation(original_transform, local_rotation_angle)

print("\nNew transformation matrix after local x-axis rotation:")
print(new_transform)

# Extract the translation and rotation from the new transformation
new_translation = new_transform[:3, 3]
new_rotation_matrix = new_transform[:3, :3]

# Convert back to quaternion and Euler angles
new_quaternion = rotation_matrix_to_quaternion(new_rotation_matrix)
new_euler = rotation_matrix_to_euler(new_rotation_matrix)

print("\nNew translation:")
print(f"({new_translation[0]:.6f}, {new_translation[1]:.6f}, {new_translation[2]:.6f})")

print("\nNew rotation as quaternion (x, y, z, w):")
print(f"({new_quaternion[0]:.6f}, {new_quaternion[1]:.6f}, {new_quaternion[2]:.6f}, {new_quaternion[3]:.6f})")

print("\nNew rotation as Euler angles (radians):")
print(f"({new_euler[0]:.6f}, {new_euler[1]:.6f}, {new_euler[2]:.6f})")

print("\nNew rotation as Euler angles (degrees):")
print(f"({np.degrees(new_euler[0]):.6f}, {np.degrees(new_euler[1]):.6f}, {np.degrees(new_euler[2]):.6f})")

# Calculate the equivalent static_transform_publisher command
print("\nEquivalent static_transform_publisher command:")
print(f"ros2 run tf2_ros static_transform_publisher {new_translation[0]:.6f} {new_translation[1]:.6f} {new_translation[2]:.6f} {new_euler[0]:.6f} {new_euler[1]:.6f} {new_euler[2]:.6f} ur10_base_link camera_link")