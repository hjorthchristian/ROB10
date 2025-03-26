import numpy as np

tf_base_optical = np.array([[0.9955 , 0.095 , 0.0073, 0.4989],
                            [0.095 , -0.9955 , -0.0053, 0.0653],
                            [0.0067, 0.006 , -1.0000, 1.1127],
                            [0.0000, 0.0000, 0.0000, 1.0000]])

#optical_camera_link  0.002 -1.000  0.004 -0.059
tf_optical_camera = np.array([[0.002, -1.000, 0.004, -0.059],
                              [0.005, -0.004, -1.000, -0.000],
                              [1.000, 0.002, 0.005, 0.001],
                              [0.000, 0.000, 0.000, 1.000]])

tf_base_camera = tf_base_optical @ tf_optical_camera
print(tf_base_camera)

def normalize_rotation_matrix(R):
    """
    Orthonormalize a 3x3 matrix to ensure it's a valid rotation matrix.
    Uses Singular Value Decomposition (SVD) to find the closest orthogonal matrix.
    
    Args:
        R (numpy.ndarray): 3x3 matrix to normalize
        
    Returns:
        numpy.ndarray: 3x3 orthonormal rotation matrix
    """
    U, _, Vh = np.linalg.svd(R)
    return U @ Vh

def rotation_matrix_to_rpy(R):
    """
    Convert a 3x3 rotation matrix to roll-pitch-yaw angles (in radians).
    
    Args:
        R (numpy.ndarray): 3x3 rotation matrix
    
    Returns:
        tuple: (roll, pitch, yaw) in radians
    
    Note:
        Uses the convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        This is the ZYX convention, common in robotics and aerospace.
    """
    # Check if the input is a valid rotation matrix
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix")
    
    # Normalize the rotation matrix to ensure it's orthonormal
    R = normalize_rotation_matrix(R)
    
    # Handle the singularity case (gimbal lock) when pitch is ±90°
    # This happens when R[2,0] = ±1
    if abs(R[2, 0]) >= 1 - 1e-6:
        # Gimbal lock case
        yaw = 0  # Set yaw to 0 as a convention
        
        if R[2, 0] < 0:  # pitch = -90°
            pitch = np.pi / 2
            roll = yaw + np.arctan2(R[0, 1], R[0, 2])
        else:  # pitch = 90°
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])
    else:
        # Regular case
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    
    # Convert to degrees for easier reading
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    return roll, pitch, yaw, roll_deg, pitch_deg, yaw_deg

# Extract rotation matrix from transformation matrix
rotation_matrix = tf_base_camera[:3, :3]
print("Determinant before normalization:", np.linalg.det(rotation_matrix))

# Get RPY angles
roll, pitch, yaw, roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_rpy(rotation_matrix)

print("\nRoll (rad):", roll)
print("Pitch (rad):", pitch)
print("Yaw (rad):", yaw)
print("\nRoll (deg):", roll_deg)
print("Pitch (deg):", pitch_deg)
print("Yaw (deg):", yaw_deg)

# Also print translation vector
translation = tf_base_camera[:3, 3]
print("\nTranslation (x, y, z):", translation)