import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# ChArUco board setup with updated parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
detector_params = cv2.aruco.DetectorParameters()
board = cv2.aruco.CharucoBoard(
    size=(4, 3),
    squareLength=0.068,
    markerLength=0.045,
    dictionary=dictionary
)

# Create detectors with proper parameter handling
aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
charuco_detector = cv2.aruco.CharucoDetector(
    board=board,
    charucoParams=cv2.aruco.CharucoParameters(),
    detectorParams=detector_params,
    refineParams=cv2.aruco.RefineParameters()  # Added refinement parameters
)

# Camera intrinsics (verify distortion model compatibility)
camera_matrix = np.array([
    [644.73888997, 0.00000000e+00, 650.03284347],
    [0.00000000e+00,  644.73888997, 361.77313771],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
# Updated distortion coefficients format for OpenCV 4.11
dist_coeffs = np.array([[-0.06254493, 0.07836239, 0.00043746, 0.00122365, -0.05503944]])

def detect_charuco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect markers using class-based detector
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        # Detect ChArUco board with refined parameters
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
            gray,
            markerCorners=corners,
            markerIds=ids
        )
        
        if charuco_corners is not None and len(charuco_corners) >= 4:
            # Get 3D points using board methods
            obj_points = board.getChessboardCorners()[charuco_ids.flatten()]
            
            # Pose estimation with updated solvePnP parameters
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                charuco_corners.astype(np.float32),
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE  # Recommended for ChArUco
            )
            if success:
                return rvec, tvec
    return None, None

def main():
    # FIXED: Changed to store gripper-to-base transformations
    R_gripper2base = []  # Correctly named for gripper-to-base transformation
    t_gripper2base = []  # Correctly named for gripper-to-base transformation
    R_target2cam = []
    t_target2cam = []

    # Data collection with enhanced error handling
    valid_frames = 0
    for i in range(1, 101):
        image_path = f'calibration_data/frame_{i:03d}.png'
        transform_path = f'calibration_data/transform_{i:03d}.txt'
        
        if not (os.path.exists(image_path) and os.path.exists(transform_path)):
            print(f"Stopped at frame {i}: files not found")
            break

        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            
            rvec, tvec = detect_charuco(image)
            if rvec is None or tvec is None:
                print(f"Skipping frame {i}: ChArUco detection failed")
                continue

            # Store camera pose
            R_cam, _ = cv2.Rodrigues(rvec)
            R_target2cam.append(R_cam)
            t_target2cam.append(tvec)

            # Parse robot transform file with flexible format handling
            with open(transform_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                
                # Flexible translation parsing
                t_line = next((l for l in lines if 'translation' in l.lower()), None)
                if t_line:
                    t_values = t_line.split()[1:]  # Skip label
                    try:
                        t = np.array([float(x) for x in t_values[:3]]).reshape(3,1)
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing translation in {transform_path}: {e}")
                        continue
                else:
                    print(f"No translation found in {transform_path}")
                    continue

                # Flexible rotation parsing
                r_line = next((l for l in lines if 'rotation' in l.lower()), None)
                if r_line:
                    r_values = r_line.split()[1:]  # Skip label
                    try:
                        if len(r_values) == 4:  # Quaternion input
                            q = [float(x) for x in r_values]
                            R_robot = R.from_quat(q).as_matrix()
                        else:  # Assume rotation vector
                            rvec_robot = np.array([float(x) for x in r_values[:3]])
                            R_robot, _ = cv2.Rodrigues(rvec_robot)
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing rotation in {transform_path}: {e}")
                        continue
                else:
                    print(f"No rotation found in {transform_path}")
                    continue

                # FIXED: Properly invert the transformation to get gripper-to-base
                # The transform files have base-to-gripper, but we need gripper-to-base
                R_gripper2base_mat = R_robot.T  # Transpose gives inverse rotation
                t_gripper2base_mat = -np.dot(R_robot.T, t)  # Proper translation inversion
                
                R_gripper2base.append(R_gripper2base_mat)
                t_gripper2base.append(t_gripper2base_mat)
                valid_frames += 1

        except Exception as e:
            print(f"Error processing frame {i}: {str(e)}")
            continue

    if valid_frames >= 5:
        print(f"\nSuccessfully processed {valid_frames} valid pose pairs")
        
        # FIXED: Try multiple calibration methods to find the best result
        calibration_methods = {
            'TSAI': cv2.CALIB_HAND_EYE_TSAI,
            'PARK': cv2.CALIB_HAND_EYE_PARK,
            'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
            'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF  # Added alternative method
        }
        
        for method_name, method in calibration_methods.items():
            try:
                print(f"\nAttempting calibration with method: {method_name}")
                R_cam2base, t_cam2base = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method
                )
                
                print(f"\nHand-Eye Calibration Result ({method_name}):")
                print("Camera to Base Rotation Matrix:")
                print(np.round(R_cam2base, 4))
                print("\nCamera to Base Translation (meters):")
                print(np.round(t_cam2base, 4))
                
                # Also display in centimeters for easier comparison with expected values
                print("\nCamera to Base Translation (centimeters):")
                print(np.round(t_cam2base * 100, 4))
                
                # Print Euler angles for easier interpretation
                euler_angles = R.from_matrix(R_cam2base).as_euler('xyz', degrees=True)
                print("\nCamera to Base Euler Angles (degrees, XYZ):")
                print(np.round(euler_angles, 4))
                
            except Exception as e:
                print(f"\nCalibration failed with method {method_name}: {str(e)}")
    else:
        print(f"\nInsufficient data: Only {valid_frames} valid pairs (minimum 5 required)")


if __name__ == "__main__":
    main()