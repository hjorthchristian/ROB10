import numpy as np

# Camera intrinsics (example values)
fx = 525.0  # focal length x
fy = 525.0  # focal length y
cx = 319.5  # principal point x
cy = 239.5  # principal point y

# Example 2D image coordinates (u, v) for the four corners of the top face
points2d = np.array([
    [100, 150],
    [200, 150],
    [200, 250],
    [100, 250]
])

# Corresponding depth values (in meters) for each point
depths = np.array([1.0, 1.0, 1.0, 1.0])  # assume constant depth for simplicity

#Convert 2D (with depth) to 3D coordinates using the pinhole model.
# x = (u - cx)*d/fx, y = (v - cy)*d/fy, z = d.
points3d = []
for (u, v), d in zip(points2d, depths):
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d
    points3d.append([x, y, z])
points3d = np.array(points3d)

# Fit a plane to these 3D points using SVD.
# Compute the centroid of the points.
centroid = np.mean(points3d, axis=0)

# Center the points.
pts_centered = points3d - centroid

# Singular Value Decomposition.
U, S, Vt = np.linalg.svd(pts_centered)
# The plane's normal is the singular vector corresponding to the smallest singular value.
normal = Vt[-1]  # This is the unit normal vector of the plane.

# Define a coordinate frame on the plane.
# Choose an in-plane vector (e.g., from the centroid to the first corner).
vec = points3d[0] - centroid
# Project this vector onto the plane to ensure it is in-plane.
vec_proj = vec - np.dot(vec, normal) * normal
x_axis = vec_proj / np.linalg.norm(vec_proj)

# Define the y-axis as the cross product of the plane normal and the x-axis.
y_axis = np.cross(normal, x_axis)
y_axis = y_axis / np.linalg.norm(y_axis)

# Compose the rotation matrix. Here, we set the plane's coordinate frame as:
# x-axis, y-axis, and z-axis (which is the plane normal).
R = np.stack((x_axis, y_axis, normal), axis=1)  # each column is an axis

# Create the final 4x4 transformation matrix.
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = centroid

print("Estimated pose (4x4 transformation matrix):")
print(T)
