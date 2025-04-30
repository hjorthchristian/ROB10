import numpy as np

# Define test point in camera frame
point = np.array([-0.064459, -0.689965, 1.633000, 1.0])  # Your point with homogeneous coordinate X_CAM: -0.064459 m
# Y_CAM: -0.689965 m
# Z_CAM: 1.633000 m
print("Original point in camera frame:", point[:3])

T_base_to_camera = np.array([
    [0.995, 0.095, 0.007, 0.499],
    [0.095, -0.995, -0.005, 0.065],
    [0.006, 0.006, -1.000, 1.114],
    [0.000, 0.000, 0.000, 1.000]
])

# Transform point from base to camera frame
point_in_camera = T_base_to_camera @ point

print("\nTransformed point in camera frame:", point_in_camera[:3])

# Main transform from camera_color_optical_frame to ur10_base_link
transform_matrix = np.array([
    [0.995, 0.095, 0.007, -0.510],
    [0.095, -0.995, -0.005, 0.010],
    [0.006, 0.006, -1.000, 1.110],
    [0.000, 0.000, 0.000, 1.000]
])

# Define the expected point range based on user input
expected_point_min = np.array([0.3, 0.77, -0.75])
expected_point_max = np.array([0.44, 0.67, -0.45])
expected_point_center = (expected_point_min + expected_point_max) / 2
print(f"Expected point range: x: {expected_point_min[0]} to {expected_point_max[0]}, " +
      f"y: {expected_point_min[1]} to {expected_point_max[1]}, " +
      f"z: {expected_point_min[2]} to {expected_point_max[2]}")

# Function to calculate distance to expected point range
def distance_to_expected(point_coords):
    """Calculate how close a point is to being within the expected range"""
    # For each dimension, calculate distance to the range
    distances = []
    for i in range(3):
        if point_coords[i] < expected_point_min[i]:
            distances.append(expected_point_min[i] - point_coords[i])
        elif point_coords[i] > expected_point_max[i]:
            distances.append(point_coords[i] - expected_point_max[i])
        else:
            distances.append(0)  # Point is within range for this dimension
            
    # Return both the Euclidean distance and the individual dimension distances
    return np.sqrt(sum(d*d for d in distances)), distances

# Function to create rotation matrices for all 90-degree rotations
def create_90_degree_rotations():
    """Generate all possible rotation matrices with 90-degree increments"""
    rotations = []
    
    # Create basic rotation matrices (90-degree rotations around X, Y, Z)
    def rot_x(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    
    def rot_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
    
    def rot_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    # Generate all combinations of rotations in 90-degree increments
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]  # 0, 90, 180, 270 degrees
    
    # Create all combinations of rotations
    for x_angle in angles:
        for y_angle in angles:
            for z_angle in angles:
                # Apply rotations in different orders (XYZ, XZY, YXZ, YZX, ZXY, ZYX)
                # Order XYZ
                r_xyz = rot_z(z_angle) @ rot_y(y_angle) @ rot_x(x_angle)
                rotations.append((r_xyz, f"X({int(x_angle*180/np.pi)}°) Y({int(y_angle*180/np.pi)}°) Z({int(z_angle*180/np.pi)}°)"))
                
                # Order XZY
                r_xzy = rot_y(y_angle) @ rot_z(z_angle) @ rot_x(x_angle)
                rotations.append((r_xzy, f"X({int(x_angle*180/np.pi)}°) Z({int(z_angle*180/np.pi)}°) Y({int(y_angle*180/np.pi)}°)"))
                
                # Order YXZ
                r_yxz = rot_z(z_angle) @ rot_x(x_angle) @ rot_y(y_angle)
                rotations.append((r_yxz, f"Y({int(y_angle*180/np.pi)}°) X({int(x_angle*180/np.pi)}°) Z({int(z_angle*180/np.pi)}°)"))
                
                # Order YZX
                r_yzx = rot_x(x_angle) @ rot_z(z_angle) @ rot_y(y_angle)
                rotations.append((r_yzx, f"Y({int(y_angle*180/np.pi)}°) Z({int(z_angle*180/np.pi)}°) X({int(x_angle*180/np.pi)}°)"))
                
                # Order ZXY
                r_zxy = rot_y(y_angle) @ rot_x(x_angle) @ rot_z(z_angle)
                rotations.append((r_zxy, f"Z({int(z_angle*180/np.pi)}°) X({int(x_angle*180/np.pi)}°) Y({int(y_angle*180/np.pi)}°)"))
                
                # Order ZYX
                r_zyx = rot_x(x_angle) @ rot_y(y_angle) @ rot_z(z_angle)
                rotations.append((r_zyx, f"Z({int(z_angle*180/np.pi)}°) Y({int(y_angle*180/np.pi)}°) X({int(x_angle*180/np.pi)}°)"))
    
    return rotations

# Function to format the mapping of axes
def explain_mapping(matrix):
    """Explains which camera axis maps to which robot axis in the rotation matrix"""
    mappings = []
    # Round for cleaner display
    rounded = np.round(matrix[:3,:3], 2)
    
    for i, row in enumerate(rounded):
        dest_axis = ["X", "Y", "Z"][i]
        mapping = f"Robot {dest_axis} ← "
        
        # Check if this is a simple mapping or a combination
        simple_mapping = False
        for j, val in enumerate(row):
            if abs(abs(val) - 1.0) < 0.1:  # Close to 1 or -1
                source_axis = ["X", "Y", "Z"][j]
                sign = "" if val > 0 else "-"
                mapping += f"{sign}Camera {source_axis}"
                simple_mapping = True
                break
        
        # If not a simple mapping, show the combination
        if not simple_mapping:
            terms = []
            for j, val in enumerate(row):
                if abs(val) > 0.1:  # Non-negligible value
                    source_axis = ["X", "Y", "Z"][j]
                    sign = "+" if val > 0 else "-"
                    if j > 0 and sign == "+":
                        terms.append(f"+{val:.2f}×Camera {source_axis}")
                    else:
                        terms.append(f"{val:.2f}×Camera {source_axis}")
            mapping += " ".join(terms)
        
        mappings.append(mapping)
    
    return "\n".join(mappings)

# Apply the standard transform directly for reference
direct_result = transform_matrix @ point
print("\nDirect transform result:", direct_result[:3])
print("Distance to expected:", distance_to_expected(direct_result[:3])[0])

# Generate and test all 90-degree rotations
rotations = create_90_degree_rotations()
print(f"\nTesting {len(rotations)} different 90-degree rotation combinations...")

# Store results for ranking
results = []

for i, (rot_matrix, rot_name) in enumerate(rotations):
    # Apply rotation and transform
    combined = transform_matrix @ rot_matrix
    result = combined @ point
    result_coords = result[:3]
    
    # Calculate distance to expected point
    dist, dim_dists = distance_to_expected(result_coords)
    
    # Store the result
    results.append({
        'id': i+1,
        'rotation': rot_matrix,
        'rot_name': rot_name,
        'result': result_coords,
        'distance': dist,
        'dim_distances': dim_dists
    })

# Sort results by distance to expected point
results.sort(key=lambda x: x['distance'])

# Display the top 10 closest matches
print("\n=== TOP 10 CLOSEST MATCHES TO EXPECTED POINT ===")
for i, result in enumerate(results[:10]):
    print(f"\nRank #{i+1}: Distance to target = {result['distance']:.4f}")
    print(f"Rotation: {result['rot_name']}")
    print("Mapping:")
    print(explain_mapping(result['rotation']))
    print(f"Result point: [{result['result'][0]:.4f}, {result['result'][1]:.4f}, {result['result'][2]:.4f}]")
    print(f"Dimension distances: X:{result['dim_distances'][0]:.4f}, Y:{result['dim_distances'][1]:.4f}, Z:{result['dim_distances'][2]:.4f}")
    print("Rotation matrix:")
    print(np.round(result['rotation'][:3,:3], 2))
    print("-" * 60)

# Also test the commonly used RealSense mappings
print("\n=== COMMON REALSENSE MAPPINGS ===")

# Create a list of common RealSense mappings
realsense_mappings = [
    # Standard RealSense mapping
    (np.array([
        [0, 0, 1, 0],   # Camera Z → Robot X (depth becomes forward)
        [-1, 0, 0, 0],  # Camera -X → Robot Y (right becomes left)
        [0, -1, 0, 0],  # Camera -Y → Robot Z (down becomes up)
        [0, 0, 0, 1]
    ]), "Standard RealSense (Z→X, -X→Y, -Y→Z)"),
    
    # Alternative with no Y inversion
    (np.array([
        [0, 0, 1, 0],   # Camera Z → Robot X
        [-1, 0, 0, 0],  # Camera -X → Robot Y
        [0, 1, 0, 0],   # Camera Y → Robot Z (no inversion)
        [0, 0, 0, 1]
    ]), "RealSense Alt 1 (Z→X, -X→Y, Y→Z)"),
    
    # Alternative with no X inversion
    (np.array([
        [0, 0, 1, 0],   # Camera Z → Robot X
        [1, 0, 0, 0],   # Camera X → Robot Y (no inversion)
        [0, -1, 0, 0],  # Camera -Y → Robot Z
        [0, 0, 0, 1]
    ]), "RealSense Alt 2 (Z→X, X→Y, -Y→Z)"),
    
    # Alternative with no inversions
    (np.array([
        [0, 0, 1, 0],   # Camera Z → Robot X
        [1, 0, 0, 0],   # Camera X → Robot Y
        [0, 1, 0, 0],   # Camera Y → Robot Z
        [0, 0, 0, 1]
    ]), "RealSense Alt 3 (Z→X, X→Y, Y→Z)"),
    
    # X forward mapping
    (np.array([
        [1, 0, 0, 0],   # Camera X → Robot X
        [0, -1, 0, 0],  # Camera -Y → Robot Y
        [0, 0, -1, 0],  # Camera -Z → Robot Z
        [0, 0, 0, 1]
    ]), "RealSense Alt 4 (X→X, -Y→Y, -Z→Z)"),
    
    # Y forward mapping
    (np.array([
        [0, 1, 0, 0],   # Camera Y → Robot X
        [-1, 0, 0, 0],  # Camera -X → Robot Y
        [0, 0, -1, 0],  # Camera -Z → Robot Z
        [0, 0, 0, 1]
    ]), "RealSense Alt 5 (Y→X, -X→Y, -Z→Z)")
]

# Test each RealSense mapping and rank them
realsense_results = []

for rot_matrix, name in realsense_mappings:
    combined = transform_matrix @ rot_matrix
    result = combined @ point
    result_coords = result[:3]
    
    dist, dim_dists = distance_to_expected(result_coords)
    
    realsense_results.append({
        'name': name,
        'rotation': rot_matrix,
        'result': result_coords,
        'distance': dist,
        'dim_distances': dim_dists
    })

# Sort RealSense results by distance
realsense_results.sort(key=lambda x: x['distance'])

# Display RealSense results
for i, result in enumerate(realsense_results):
    print(f"\nRealSense Mapping #{i+1}: {result['name']}")
    print(f"Distance to target: {result['distance']:.4f}")
    print("Mapping:")
    print(explain_mapping(result['rotation']))
    print(f"Result point: [{result['result'][0]:.4f}, {result['result'][1]:.4f}, {result['result'][2]:.4f}]")
    print(f"Dimension distances: X:{result['dim_distances'][0]:.4f}, Y:{result['dim_distances'][1]:.4f}, Z:{result['dim_distances'][2]:.4f}")
    print("Rotation matrix:")
    print(np.round(result['rotation'][:3,:3], 2))
    print("-" * 60)

# Add a new section at the end for reverse transformation
print("\n=== REVERSE TRANSFORMATION (ROBOT BASE TO CAMERA) ===")

# Calculate the inverse of the transformation matrix
def invert_transform(transform):
    """Invert a 4x4 transformation matrix"""
    # Extract the rotation matrix (3x3) and translation vector (3x1)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    
    # Calculate inverse rotation (transpose) and inverse translation
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ translation
    
    # Create the inverse transformation matrix
    inv_transform = np.eye(4)
    inv_transform[:3, :3] = inv_rotation
    inv_transform[:3, 3] = inv_translation
    
    return inv_transform

# Invert the main transformation matrix
inverse_transform = invert_transform(transform_matrix)
print("Inverse transformation matrix:")
print(np.round(inverse_transform, 3))

# Function to transform a point from robot base to camera frame
def transform_to_camera(point_base):
    """Transform a point from robot base frame to camera frame"""
    # Convert to homogeneous coordinates if needed
    if len(point_base) == 3:
        point_base = np.append(point_base, 1.0)
    
    # Apply inverse transformation
    point_camera = inverse_transform @ point_base
    
    return point_camera[:3]  # Return just the 3D coordinates

# Example: Transform the given point back to camera frame
robot_point = np.array([0.372, 0.739, -0.521, 1.0])
camera_point = transform_to_camera(robot_point)

print(f"\nPoint in robot base frame: {robot_point[:3]}")
print(f"Transformed to camera frame: {camera_point}")

# Verify by transforming back to robot frame
robot_point_verification = transform_matrix @ np.append(camera_point, 1.0)
print(f"Verification - transformed back to robot frame: {robot_point_verification[:3]}")
print(f"Original robot point: {robot_point[:3]}")
print(f"Difference: {np.round(robot_point[:3] - robot_point_verification[:3], 6)}")

# Apply the reverse transformation with each of the top rotation matrices
print("\n=== USING TOP ROTATION MATRICES FOR REVERSE TRANSFORMATION ===")

for i, result in enumerate(results[:3]):  # Show top 3 rotations
    rot_matrix = result['rotation']
    rot_name = result['rot_name']
    
    # Calculate the combined transform with this rotation
    combined = transform_matrix @ rot_matrix
    
    # Calculate the inverse of the combined transform
    inverse_combined = invert_transform(combined)
    
    # Apply the inverse transform to the robot point
    camera_point_with_rot = inverse_combined @ robot_point
    
    print(f"\nUsing rotation {rot_name}:")
    print(f"Point in camera frame: {camera_point_with_rot[:3]}")
    
    # Verify by transforming back
    verification = combined @ np.append(camera_point_with_rot[:3], 1.0)
    print(f"Verification error: {np.round(robot_point[:3] - verification[:3], 6)}")

# Add visualization using matplotlib
print("\n=== VISUALIZATION OF COORDINATE FRAMES AND POINTS ===")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Define a custom 3D arrow for better visualization
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        # Calculate the z-order based on the average z position
        avg_z = (zs[0] + zs[1]) / 2
        return avg_z  # Return the z-order

    def draw(self, renderer):
        super().draw(renderer)

# Function to plot a coordinate frame
def plot_frame(ax, origin, rotation_matrix, scale=0.1, name=""):
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    
    # Plot the origin
    ax.scatter(origin[0], origin[1], origin[2], color='black', s=50)
    
    # Plot coordinate frame text label
    ax.text(origin[0], origin[1], origin[2] + 0.05, name, fontsize=10)
    
    # Plot each axis
    for i in range(3):
        axis = rotation_matrix[:3, i] * scale
        arrow = Arrow3D([origin[0], origin[0] + axis[0]],
                        [origin[1], origin[1] + axis[1]],
                        [origin[2], origin[2] + axis[2]],
                        mutation_scale=20, lw=2, arrowstyle='-|>', color=colors[i])
        ax.add_artist(arrow)
        ax.text(origin[0] + axis[0]*1.1, 
                origin[1] + axis[1]*1.1, 
                origin[2] + axis[2]*1.1, 
                f"{name}_{labels[i]}", color=colors[i], fontsize=8)

def print_matrix(matrix):
    """Print the matrix in a formatted way"""
    for row in matrix:
        print("  ".join([f"{val:6.3f}" for val in row]))

# Original transformation matrix (base to camera optical frame)
T_base_to_camera = np.array([
    [0.995, 0.095, 0.007, 0.499],
    [0.095, -0.995, -0.005, 0.065],
    [0.006, 0.006, -1.000, 1.114],
    [0.000, 0.000, 0.000, 1.000]
])

# Compute the inverse transformation (camera optical frame to base)
T_camera_to_base = np.linalg.inv(T_base_to_camera)

print("Original Transform (Base to Camera):")
print_matrix(T_base_to_camera)
print("\nInverse Transform (Camera to Base):")
print_matrix(T_camera_to_base)

# Verify that the inverse is correct by multiplying the two matrices
verification = np.matmul(T_base_to_camera, T_camera_to_base)
print("\nVerification (should be identity matrix):")
print_matrix(verification)

# Extract position and orientation data from the inverse transform
position = T_camera_to_base[:3, 3]
rotation_matrix = T_camera_to_base[:3, :3]

print("\nCamera position in base frame:")
print(f"x: {position[0]:.3f}, y: {position[1]:.3f}, z: {position[2]:.3f}")

# You can also compute Euler angles if needed
# This is just one possible convention (ZYX)
try:
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=True)
    print("\nCamera orientation in base frame (Euler angles XYZ in degrees):")
    print(f"Roll: {euler[0]:.3f}, Pitch: {euler[1]:.3f}, Yaw: {euler[2]:.3f}")
except ImportError:
    print("\nInstall scipy to compute Euler angles")

# Create the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Camera frame origin and orientation in world coordinates
base_to_camera_transform = np.array([
    [0.995, 0.095, 0.007, 0.499],
    [0.095, -0.995, -0.005, 0.065],
    [0.006, 0.006, -1.000, 1.114],
    [0.000, 0.000, 0.000, 1.000]
])
camera_orientation = base_to_camera_transform[:3, :3]
camera_origin = np.array([0.499, 0.065, 1.114])

# Base link frame (robot base)
base_origin = np.array([0, 0, 0])
base_orientation = np.eye(3)  # Identity matrix for the base frame

# Plot the frames
plot_frame(ax, base_origin, base_orientation, scale=0.3, name="base_link")
plot_frame(ax, camera_origin, camera_orientation, scale=0.3, name="camera")

# Plot the points
# Original point in camera frame - we need to transform to world coordinates
camera_point_in_world = transform_matrix @ point
ax.scatter(camera_point_in_world[0], camera_point_in_world[1], camera_point_in_world[2], 
           color='orange', s=100, marker='o', label='Original Point (camera)')

# Expected point in base frame
ax.scatter(expected_point_center[0], expected_point_center[1], expected_point_center[2], 
           color='green', s=100, marker='o', label='Expected Point (base)')

# Direct transform result
ax.scatter(direct_result[0], direct_result[1], direct_result[2], 
           color='blue', s=100, marker='o', label='Direct Transform (base)')

# Add expected point range as a box
from itertools import product
for p in product([expected_point_min[0], expected_point_max[0]],
                 [expected_point_min[1], expected_point_max[1]],
                 [expected_point_min[2], expected_point_max[2]]):
    ax.scatter(*p, color='green', alpha=0.3, s=20)

# Connect the box points to form a cube
for s, e in [
    ([0, 0, 0], [1, 0, 0]), ([0, 0, 0], [0, 1, 0]), ([0, 0, 0], [0, 0, 1]),
    ([1, 0, 0], [1, 1, 0]), ([1, 0, 0], [1, 0, 1]), ([0, 1, 0], [1, 1, 0]),
    ([0, 1, 0], [0, 1, 1]), ([0, 0, 1], [1, 0, 1]), ([0, 0, 1], [0, 1, 1]),
    ([1, 1, 1], [1, 1, 0]), ([1, 1, 1], [1, 0, 1]), ([1, 1, 1], [0, 1, 1])
]:
    ax.plot3D([expected_point_min[0] + s[0]*(expected_point_max[0] - expected_point_min[0]),
               expected_point_min[0] + e[0]*(expected_point_max[0] - expected_point_min[0])],
              [expected_point_min[1] + s[1]*(expected_point_max[1] - expected_point_min[1]),
               expected_point_min[1] + e[1]*(expected_point_max[1] - expected_point_min[1])],
              [expected_point_min[2] + s[2]*(expected_point_max[2] - expected_point_min[2]),
               expected_point_min[2] + e[2]*(expected_point_max[2] - expected_point_min[2])],
              'green', alpha=0.3)

# Add legend, labels, and title
ax.legend()
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Camera and Base Link Coordinate Frames with Points')

# Set equal aspect ratio
max_range = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).T
max_extent = np.max(max_range[1] - max_range[0])
ax.set_box_aspect([1, 1, 1])

# Set the limits to ensure everything is visible
ax.set_xlim(min(camera_origin[0], base_origin[0], expected_point_min[0], direct_result[0]) - 0.5, 
            max(camera_origin[0], base_origin[0], expected_point_max[0], direct_result[0]) + 0.5)
ax.set_ylim(min(camera_origin[1], base_origin[1], expected_point_min[1], direct_result[1]) - 0.5, 
            max(camera_origin[1], base_origin[1], expected_point_max[1], direct_result[1]) + 0.5)
ax.set_zlim(min(camera_origin[2], base_origin[2], expected_point_min[2], direct_result[2]) - 0.5, 
            max(camera_origin[2], base_origin[2], expected_point_max[2], direct_result[2]) + 0.5)

# Add grid for better depth perception
ax.grid(True)

print("Plotting coordinate frames and points...")
plt.tight_layout()
plt.show()