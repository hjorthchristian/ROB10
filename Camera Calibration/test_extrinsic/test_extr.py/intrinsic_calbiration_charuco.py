import cv2
import numpy as np
import glob

# Define the ChArUco board parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
detector_params = cv2.aruco.DetectorParameters()
charuco_params = cv2.aruco.CharucoParameters()  # Create CharucoParameters object
board = cv2.aruco.CharucoBoard(
    size=(4, 3),
    squareLength=0.068,
    markerLength=0.045,
    dictionary=dictionary
)

# Create detector parameters
detector_params = cv2.aruco.DetectorParameters()
charuco_params = cv2.aruco.CharucoParameters()

# Create CharucoDetector with proper parameters
detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

# Collect data from each frame
all_charuco_corners = []
all_charuco_ids = []
all_image_points = []
all_object_points = []
all_images = []

# Get a list of all calibration images
images = glob.glob('calibration_data/frame_*.png')
image_size = None

for fname in images:
    # Read the image
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect ChArUco board
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    
    # If enough corners detected, add to calibration data
    if charuco_ids is not None and len(charuco_corners) > 3:
        # Get corresponding object points and image points
        current_image_points = charuco_corners
        
        # Get object points (3D coordinates of the board corners)
        current_object_points = []
        for i in range(len(charuco_ids)):
            idx = charuco_ids[i][0]
            # Get 3D coordinates from the board
            point = board.getChessboardCorners()[idx]
            current_object_points.append(point)
        
        if len(current_image_points) > 0 and len(current_object_points) > 0:
            print(f"Detected corners in {fname}")
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            all_image_points.append(np.array(current_image_points))
            all_object_points.append(np.array(current_object_points))
            all_images.append(image)
            
            # Get image size from first valid image
            if image_size is None:
                image_size = image.shape[:2][::-1]  # width, height
            
            # Optional: Draw detection for visualization
            debug_img = image.copy()
            cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
            cv2.imwrite(f"debug_{fname.split('/')[-1]}", debug_img)
        else:
            print(f"Point matching failed in {fname}")
    else:
        print(f"Not enough corners detected in {fname}")

# Setup camera matrix for calibration
calibration_flags = 0  # Define your flags
if calibration_flags & cv2.CALIB_FIX_ASPECT_RATIO:
    camera_matrix = np.eye(3, dtype=np.float64)
    camera_matrix[0, 0] = aspectRatio  # Define aspectRatio with your value
else:
    camera_matrix = None

dist_coeffs = None

# Calibrate camera using ChArUco
if len(all_object_points) > 0:
    rep_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        camera_matrix,
        dist_coeffs,
        flags=calibration_flags
    )
    
    print("Calibration complete!")
    print("Reprojection error:", rep_error)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
    
    # Save calibration results
    np.savez("calibration_results.npz", 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs,
             error=rep_error)
else:
    print("Not enough frames captured for calibration")