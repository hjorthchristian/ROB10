import cv2
import numpy as np
import glob
import os

# Charuco board configuration
squaresX = 4
squaresY = 3
squareLength = 0.063
markerLength = 0.045
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
board = cv2.aruco.CharucoBoard(
    (squaresX, squaresY),
    squareLength=squareLength,
    markerLength=markerLength,
    dictionary=dictionary
)

# Calibration parameters (your values)
calibration_data = {
    'camera_matrix': np.array([
        [644.91407239, 0.0, 650.0619773],
        [0.0, 644.91407239, 361.79122397],
        [0.0, 0.0, 1.0]
    ]),
    'dist_coeffs': np.array([
        25.6177584, -14.9234213, 0.000695935, 0.00145449,
        135.439185, 25.7752723, -14.325311, 137.556902,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
}

# Configuration
input_folder = "calibration_data"
output_folder = "undistorted_results"
os.makedirs(output_folder, exist_ok=True)

# Initialize parameters correctly
charuco_params = cv2.aruco.CharucoParameters()
detector_params = cv2.aruco.DetectorParameters()
refine_params = cv2.aruco.RefineParameters()

# Correct constructor with proper parameter order
charuco_detector = cv2.aruco.CharucoDetector(
    board,
    charucoParams=charuco_params,
    detectorParams=detector_params,
    refineParams=refine_params
)

# Create undistortion maps
sample_image = cv2.imread(os.path.join(input_folder, "frame_001.png"))
h, w = sample_image.shape[:2]

new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
    calibration_data['camera_matrix'],
    calibration_data['dist_coeffs'],
    (w, h), 1, (w, h)
)
map1, map2 = cv2.initUndistortRectifyMap(
    calibration_data['camera_matrix'],
    calibration_data['dist_coeffs'],
    None,
    new_cam_matrix,
    (w, h),
    cv2.CV_16SC2
)

for img_path in glob.glob(os.path.join(input_folder, "frame_*.png")):
    # Process image
    img = cv2.imread(img_path)
    undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    
    # Detect with modern API
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
    
    if charuco_corners is not None and len(charuco_corners) > 0:
        # Calculate reprojection error
        obj_points = board.getChessboardCorners()[charuco_ids.flatten()]
        img_points_proj, _ = cv2.projectPoints(
            obj_points,
            np.zeros(3),  # rvec
            np.zeros(3),  # tvec
            calibration_data['camera_matrix'],
            calibration_data['dist_coeffs']
        )
        error = cv2.norm(charuco_corners, img_points_proj, cv2.NORM_L2)/len(img_points_proj)
        print(f"{os.path.basename(img_path)} - Error: {error:.4f} px")

        # Draw results
        cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
    
    # Display comparison
    combined = np.hstack((img, undistorted))
    cv2.imshow("Original vs Undistorted", combined)
    key = cv2.waitKey(5000)
    if key == 27:
        break
    
    # Save result
    output_path = os.path.join(output_folder, f"undist_{os.path.basename(img_path)}")
    cv2.imwrite(output_path, undistorted)

cv2.destroyAllWindows()
