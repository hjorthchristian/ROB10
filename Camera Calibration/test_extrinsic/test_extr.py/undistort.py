import cv2
import numpy as np
import glob
import os

# Load calibration data
calib_data = np.load("calibration_results.npz")
camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

# Create output directory if it doesn't exist
output_dir = "undistorted_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all images from the calibration folder
images = glob.glob('calibration_data/frame_*.png')

for img_path in images:
    # Extract filename without path
    filename = os.path.basename(img_path)
    
    # Read the image
    img = cv2.imread(img_path)
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Calculate optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1)
    
    # Undistort the image
    undistorted_img = cv2.undistort(
        img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Optional: crop the undistorted image to remove black borders
    # x, y, w, h = roi
    # undistorted_img = undistorted_img[y:y+h, x:x+w]
    
    # Save the undistorted image
    output_path = os.path.join(output_dir, f"undist_{filename}")
    cv2.imwrite(output_path, undistorted_img)
    
    # Create a side-by-side comparison
    comparison = np.hstack((img, undistorted_img))
    comparison_path = os.path.join(output_dir, f"compare_{filename}")
    cv2.imwrite(comparison_path, comparison)
    
    print(f"Processed {filename}")

print(f"All images undistorted. Results saved in {output_dir}/")