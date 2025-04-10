import numpy as np

# Define test point in camera frame
point = np.array([0.31809, -0.21709, 1.048, 1.0])  # Your point with homogeneous coordinate
print("Original point in camera frame:", point[:3])

# Main transform from camera_color_optical_frame to ur10_base_link
transform_matrix = np.array([
    [0.995, 0.095, 0.007, -0.510],
    [0.095, -0.995, -0.005, 0.010],
    [0.006, 0.006, -1.000, 1.110],
    [0.000, 0.000, 0.000, 1.000]
])

# Define the expected point range based on user input
expected_point_min = np.array([-0.8, -0.3, -0.05])
expected_point_max = np.array([-0.8, -0.1, 0.10])
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
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]  # 0, 90, 180, 270 degrees
    
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