import numpy as np
import cv2
import pyrealsense2 as rs

# Define the ChArUco board parameters (match charuco_calibration.py)
squaresX = 4
squaresY = 3
squareLength = 0.063  # in meters
markerLength = 0.045  # in meters

# Create dictionary and board
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
board = cv2.aruco.CharucoBoard(
    size=(squaresX, squaresY),
    squareLength=squareLength,
    markerLength=markerLength,
    dictionary=dictionary
)

# Create detector objects
aruco_detector = cv2.aruco.ArucoDetector(dictionary)
charuco_detector = cv2.aruco.CharucoDetector(board)

# Lists to store object points and image points
all_charuco_corners = []
all_charuco_ids = []

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print("Press SPACE to capture a frame")
print("Press ESC to finish and calculate camera parameters")

try:
    frame_count = 0
    while True:
        # Get frameset of color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        img = np.asanyarray(color_frame.get_data())
        
        # Detect ArUco markers
        marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(img)
        
        # Draw detected markers
        img_display = img.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(img_display, marker_corners, marker_ids)
            # Only draw the markers, don't detect ChArUco board until SPACE is pressed
        
        # Display the image
        cv2.imshow("ChArUco Calibration", img_display)
        key = cv2.waitKey(1)
        
        # Capture frame on spacebar press
        if key == ord(' '):
            if marker_ids is not None and len(marker_ids) > 0:
                # Check if ChArUco corners were found
                result = charuco_detector.detectBoard(
                    image=img,
                    markerCorners=marker_corners,
                    markerIds=marker_ids
                )
                
                # Unpack the result correctly (it may return more than 2 values)
                if len(result) >= 2:
                    charuco_corners = result[0]
                    charuco_ids = result[1]
                    
                    # Draw detected corners for visual feedback
                    cv2.aruco.drawDetectedCornersCharuco(
                        img_display, 
                        charuco_corners, 
                        charuco_ids, 
                        (0, 255, 0)
                    )
                    cv2.imshow("Captured Frame", img_display)
                    cv2.waitKey(500)  # Show for half a second
                    
                    if charuco_corners is not None and len(charuco_corners) > 4:
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)
                        frame_count += 1
                        print(f"Frame {frame_count} captured! ({len(charuco_corners)} corners)")
                    else:
                        print("Not enough ChArUco corners detected!")
                else:
                    print("ChArUco board detection failed!")
            else:
                print("No markers detected!")
        
        # Exit on ESC
        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# Perform camera calibration if we have enough frames
if len(all_charuco_corners) > 0:
    print(f"Calibrating camera using {len(all_charuco_corners)} frames...")
    
    # Get image size
    img_height, img_width = img.shape[:2]
    
    # Prepare calibration data
    obj_points = []
    img_points = []
    
    # Get object points from the board
    obj_p = board.getChessboardCorners()
    
    # Process each captured frame
    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        # Match corners with their IDs
        for i, corner_id in enumerate(ids.flatten()):
            img_points.append(corners[i])
            obj_points.append(obj_p[corner_id])
    
    # Convert to numpy arrays
    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [obj_points], [img_points], (img_width, img_height), None, None
    )
    
    # Save calibration data
    np.savez("new_camera_calibration_charuco.npz", 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs, 
             rvecs=rvecs, 
             tvecs=tvecs)
    
    print("Calibration successful!")
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"\nTotal reprojection error: {mean_error/len(obj_points)}")
else:
    print("No frames were captured! Cannot calibrate.")
