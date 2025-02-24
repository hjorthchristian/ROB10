import cv2
import numpy as np


object_points = np.array([
    [0.0,    0.0,    0.0],         # top-left corner
    [0.3,    0.0,    0.0],         # top-right corner
    [0.3,    0.3,    0.0],         # bottom-right corner
    [0.0,    0.3,    0.0]          # bottom-left corner
], dtype=np.float32)

# Example corresponding 2D image points detected in the RGB image (in pixels)
image_points = np.array([
    [320, 240],   # detected top-left
    [420, 235],   # detected top-right
    [425, 335],   # detected bottom-right
    [315, 340]    # detected bottom-left
], dtype=np.float32)

# Camera intrinsic matrix (example values)
fx = 600  # focal length in x
fy = 600  # focal length in y
cx = 320  # principal point x
cy = 240  # principal point y
camera_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)

# Assuming zero distortion (or use your calibrated distortion coefficients)
dist_coeffs = np.zeros((4, 1))

# Use solvePnPRansac to compute the pose
# Returns the rotation vector (rvec) and translation vector (tvec)
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    object_points, image_points, camera_matrix, dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=8.0, confidence=0.99, iterationsCount=100
)

if success:
    print("Rotation vector (rvec):")
    print(rvec)
    print("Translation vector (tvec):")
    print(tvec)
    print("Inliers:", inliers.ravel())
    
    # Convert rotation vector to a rotation matrix:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    print("Rotation matrix:")
    print(rotation_matrix)
else:
    print("PnP failed to find a valid solution.")
