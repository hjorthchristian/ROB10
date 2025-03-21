import pyrealsense2 as rs
import cv2
import numpy as np
import time

# Updated ChArUco board parameters
CHARUCO_SIZE = (4, 3)      # (width, height) in squares
SQUARE_LENGTH = 0.068      # meters (between chessboard corners)
MARKER_LENGTH = 0.045      # meters (actual marker size)
ARUCO_DICT = cv2.aruco.DICT_5X5_250  # Matches your dictionary

# Camera intrinsics from calibration
camera_matrix = np.array([
    [644.73888997, 0.00000000e+00, 650.03284347],
    [0.00000000e+00, 644.73888997, 361.77313771],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

dist_coeffs = np.array([
    [20.4396465, 2.98876662, 0.000683674, 0.00144313,
     166.182592, 20.5857413, 3.32463569, 169.730822,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
], dtype=np.float32)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Create ChArUco detector with optimized parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard(CHARUCO_SIZE, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
charuco_detector = cv2.aruco.CharucoDetector(
    board=board,
    detectorParams=detector_params,
    refineParams=cv2.aruco.RefineParameters()
)

# Visualization settings
last_update = time.time()
current_translation = None
window_name = 'ChArUco Pose Detection'

def validate_detection(corners, ids):
    """Enhanced validation for board detection reliability"""
    if ids is None or len(ids) < 3:  # Minimum 3 markers for stability
        return False
    
    # Check geometric consistency
    perimeter = cv2.arcLength(corners, True)
    if perimeter < 100:  # Minimum perimeter threshold in pixels
        return False
        
    return True

try:
    pipeline.start(config)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        print("Starting")
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        debug_image = color_image.copy()
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Detect with subpixel refinement
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        
        if validate_detection(charuco_corners, charuco_ids):
            # Convert detection results for solvePnP
            obj_points = board.getChessboardCorners()[charuco_ids.flatten()]
            img_points = charuco_corners.reshape(-1, 2).astype(np.float32)

            # Solve PnP with rational distortion model
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Update translation every 5 seconds
                if time.time() - last_update >= 5:
                    current_translation = tvec.flatten() * 1000  # Convert to mm
                    last_update = time.time()
                    print(f"Translation (mm): X: {current_translation[0]:.1f}, "
                          f"Y: {current_translation[1]:.1f}, Z: {current_translation[2]:.1f}")

                # Visualize with dynamic axis scaling
                axis_scale = 0.15 * np.linalg.norm(tvec)  # Auto-scaling
                cv2.drawFrameAxes(debug_image, camera_matrix, dist_coeffs, 
                                rvec, tvec, axis_scale)
                
                # Draw detected markers
                debug_image = cv2.aruco.drawDetectedCornersCharuco(
                    debug_image, charuco_corners, charuco_ids
                )

        # Display overlay
        if current_translation is not None:
            text = (f"X: {current_translation[0]:.1f}mm | "
                    f"Y: {current_translation[1]:.1f}mm | "
                    f"Z: {current_translation[2]:.1f}mm")
            cv2.putText(debug_image, text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, 
                        cv2.LINE_AA)

        cv2.imshow(window_name, debug_image)
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
