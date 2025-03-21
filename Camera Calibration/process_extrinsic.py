#!/usr/bin/env python3

import cv2
import numpy as np
import os
import glob
import re
import transformations as tf  # For handling quaternions and transformations

def parse_transform_file(file_path):
    """
    Parse the transform file to extract translation and rotation data.
    
    Args:
        file_path: Path to the transform file
        
    Returns:
        A 4x4 transformation matrix
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Extract translation from first line
    trans_line = lines[0].strip()
    translation = np.array([float(x) for x in trans_line.split(':')[1].strip().split()])
    
    # Extract quaternion from second line
    rot_line = lines[1].strip()
    quaternion = np.array([float(x) for x in rot_line.split(':')[1].strip().split()])
    
    # Convert quaternion to rotation matrix
    # Format is [x, y, z, w] (ROS standard)
    quat = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])  # Convert to [x, y, z, w]
    rot_matrix = tf.quaternion_matrix(quat)
    
    # Create the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_matrix[:3, :3]
    transform_matrix[:3, 3] = translation
    
    return transform_matrix

def extract_number_from_filename(filename):
    """Extract the number from a filename like frame_001.png"""
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0

def detect_charuco(image, board, detector_params=None):
    """
    Detect Charuco board in the given image.
    
    Args:
        image: Input image
        board: CharucoBoard instance
        detector_params: Optional detector parameters
        
    Returns:
        Tuple of (corners, ids, rejectedImgPoints)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create CharucoDetector
    charuco_detector = cv2.aruco.CharucoDetector(board)
    
    # Detect Charuco board directly
    charucoCorners, charucoIds, markerCorners, markerIds = charuco_detector.detectBoard(gray)
    
    if charucoCorners is not None and len(charucoCorners) > 0 and charucoIds is not None and len(charucoIds) > 0:
        return charucoCorners, charucoIds, True
    
    return None, None, False

def calibrate_camera(image_folder, board, pattern="frame_*.png"):
    """
    Calibrate camera using Charuco boards.
    
    Args:
        image_folder: Folder containing calibration images
        board: CharucoBoard instance
        pattern: Glob pattern for image files
        
    Returns:
        Camera matrix, distortion coefficients, rotation vectors, translation vectors
    """
    # Prepare object points (3D points in board coordinate system)
    all_corners = []
    all_ids = []
    all_obj_points = []
    image_size = None
    
    # Get all calibration images
    image_files = sorted(glob.glob(os.path.join(image_folder, pattern)))
    
    if not image_files:
        raise ValueError(f"No images found in {image_folder} with pattern {pattern}")
    
    # Process each image
    for image_file in image_files:
        print(f"Processing {image_file}")
        image = cv2.imread(image_file)
        
        if image is None:
            print(f"Failed to read {image_file}")
            continue
        
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        
        # Detect Charuco board
        corners, ids, ret = detect_charuco(image, board)
        
        if ret and corners is not None and ids is not None and len(corners) > 0:
            # Get object points for these corners
            obj_points = []
            for i in range(len(ids)):
                idx = ids[i][0]
                # Get 3D coordinates of the corners in the board coordinate system
                corner_coords = board.getChessboardCorners()[idx]
                obj_points.append(corner_coords)
            
            all_corners.append(corners)
            all_ids.append(ids)
            all_obj_points.append(np.array(obj_points, dtype=np.float32))
        else:
            print(f"No Charuco board detected in {image_file}")
    
    if not all_corners:
        raise ValueError("No Charuco corners detected in any of the images")
    
    # Calibrate camera
    flags = (
        cv2.CALIB_RATIONAL_MODEL +  # Use rational model for distortion
        cv2.CALIB_FIX_ASPECT_RATIO   # Fix aspect ratio
    )
    
    camera_matrix_init = np.array([
        [1000.0, 0.0, image_size[0]/2],
        [0.0, 1000.0, image_size[1]/2],
        [0.0, 0.0, 1.0]
    ])
    
    dist_coeffs_init = np.zeros((5, 1))
    
    # Run standard OpenCV calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=all_obj_points,
        imagePoints=all_corners,
        imageSize=image_size,
        cameraMatrix=camera_matrix_init,
        distCoeffs=dist_coeffs_init,
        flags=flags
    )
    
    print(f"Camera calibration complete. Reprojection error: {ret}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, image_files

def compute_board_poses(image_folder, transform_folder, board, camera_matrix, dist_coeffs, pattern="frame_*.png"):
    """
    Compute the poses of the board and corresponding robot transforms.
    
    Args:
        image_folder: Folder containing calibration images
        transform_folder: Folder containing transform files
        board: CharucoBoard instance
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        pattern: Glob pattern for image files
        
    Returns:
        List of board poses and corresponding robot transforms
    """
    # Get all calibration images
    image_files = sorted(glob.glob(os.path.join(image_folder, pattern)))
    
    if not image_files:
        raise ValueError(f"No images found in {image_folder} with pattern {pattern}")
    
    board_poses = []  # Camera to board transforms
    robot_poses = []  # Base to end-effector transforms
    
    for image_file in image_files:
        # Extract the number to find the corresponding transform file
        num = extract_number_from_filename(image_file)
        transform_file = os.path.join(transform_folder, f"transform_{num:03d}.txt")
        
        if not os.path.exists(transform_file):
            print(f"Transform file {transform_file} not found for {image_file}")
            continue
        
        # Read the image
        image = cv2.imread(image_file)
        
        if image is None:
            print(f"Failed to read {image_file}")
            continue
        
        # Detect Charuco board
        corners, ids, ret = detect_charuco(image, board)
        
        if ret and corners is not None and ids is not None and len(corners) > 0:
            # Get object points for these corners
            obj_points = []
            for i in range(len(ids)):
                idx = ids[i][0]
                # Get 3D coordinates of the corners in the board coordinate system
                corner_coords = board.getChessboardCorners()[idx]
                obj_points.append(corner_coords)
            
            # Convert to numpy arrays of correct shape
            obj_points = np.array(obj_points, dtype=np.float32)
            img_points = corners.reshape(-1, 2)
            
            # Estimate the pose using solvePnP
            ret, rvec, tvec = cv2.solvePnP(
                objectPoints=obj_points,
                imagePoints=img_points,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )
            
            if ret:
                # Convert rvec to rotation matrix
                rmat, _ = cv2.Rodrigues(rvec)
                
                # Create camera to board transform
                cam_to_board = np.eye(4)
                cam_to_board[:3, :3] = rmat
                cam_to_board[:3, 3] = tvec.flatten()
                
                # Read robot transform (base to end-effector)
                base_to_ee = parse_transform_file(transform_file)
                
                # Add to lists
                board_poses.append(cam_to_board)
                robot_poses.append(base_to_ee)
                
                print(f"Added poses from {image_file} and {transform_file}")
            else:
                print(f"Failed to estimate board pose in {image_file}")
        else:
            print(f"No Charuco board detected in {image_file}")
    
    return board_poses, robot_poses

def calibrate_hand_eye_base(board_poses, robot_poses):
    """
    Perform hand-eye calibration (eye-on-base variant).
    
    Args:
        board_poses: List of camera to board transforms
        robot_poses: List of base to end-effector transforms
        
    Returns:
        Transformation from robot base to camera
    """
    # Convert lists to arrays suitable for OpenCV's calibrateHandEye
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    for robot_pose in robot_poses:
        R_gripper2base.append(robot_pose[:3, :3])
        t_gripper2base.append(robot_pose[:3, 3])
    
    for board_pose in board_poses:
        R_target2cam.append(board_pose[:3, :3])
        t_target2cam.append(board_pose[:3, 3])
    
    # Calibrate using TSAI method (eye-on-base variant)
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=np.array(R_gripper2base),
        t_gripper2base=np.array(t_gripper2base),
        R_target2cam=np.array(R_target2cam),
        t_target2cam=np.array(t_target2cam),
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # Create the transformation matrix
    cam2base = np.eye(4)
    cam2base[:3, :3] = R_cam2base
    cam2base[:3, 3] = t_cam2base.flatten()
    
    return cam2base

def main():
    # Define paths
    calib_data_folder = "calibration_data"
    image_folder = calib_data_folder
    transform_folder = calib_data_folder
    
    # Define Charuco board parameters (from user's input)
    squaresX = 4
    squaresY = 3
    squareLength = 0.068  # in meters
    markerLength = 0.045  # in meters
    
    # Create dictionary and board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard(
        size=(squaresX, squaresY),
        squareLength=squareLength,
        markerLength=markerLength,
        dictionary=dictionary
    )
    
    # Step 1: Calibrate camera to get intrinsics
    print("Calibrating camera...")
    camera_matrix, dist_coeffs, rvecs, tvecs, image_files = calibrate_camera(image_folder, board)
    
    # Print camera intrinsics
    print("\nCamera Matrix (Intrinsics):")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs.T)
    
    # Step 2: Compute board poses and corresponding robot transforms
    print("\nComputing board poses and robot transforms...")
    board_poses, robot_poses = compute_board_poses(
        image_folder, transform_folder, board, camera_matrix, dist_coeffs
    )
    
    if len(board_poses) < 3:
        print(f"Warning: Only {len(board_poses)} valid poses found. Calibration may be inaccurate.")
        if len(board_poses) == 0:
            print("Calibration failed. No valid poses found.")
            return
    
    # Step 3: Perform eye-on-base calibration
    print("\nPerforming eye-on-base calibration...")
    cam2base = calibrate_hand_eye_base(board_poses, robot_poses)
    
    # Print results
    print("\nCamera to Robot Base Transformation:")
    print(cam2base)
    
    # Extract and print rotation and translation in a more readable format
    R = cam2base[:3, :3]
    t = cam2base[:3, 3]
    
    # Convert rotation matrix to quaternion [x, y, z, w]
    quat = tf.quaternion_from_matrix(cam2base)
    
    print("\nTranslation (x, y, z):")
    print(t)
    
    print("\nRotation Quaternion (x, y, z, w):")
    print(quat)
    
    # Save calibration results to a file
    np.savez("calibration_results.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             cam2base=cam2base)
    
    print("\nCalibration results saved to calibration_results.npz")
    
    # Verify calibration by projecting board points onto images
    print("\nVerifying calibration by projecting board onto images...")
    verify_calibration(image_files, board, camera_matrix, dist_coeffs, cam2base, robot_poses)

def verify_calibration(image_files, board, camera_matrix, dist_coeffs, cam2base, robot_poses):
    """
    Verify calibration by projecting board points onto images.
    
    Args:
        image_files: List of image file paths
        board: CharucoBoard instance
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        cam2base: Transform from camera to robot base
        robot_poses: List of base to end-effector transforms
    """
    # Create output folder for verification images
    output_folder = "verification_images"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get board object points (3D points in board coordinate system)
    board_points = board.getChessboardCorners()
    
    for i, image_file in enumerate(image_files):
        if i >= len(robot_poses):
            continue
            
        # Read image
        image = cv2.imread(image_file)
        
        if image is None:
            continue
        
        # Get robot pose (base to end-effector)
        base2ee = robot_poses[i]
        
        # Compute camera to board transform
        base2cam = np.linalg.inv(cam2base)
        ee2board = np.eye(4)  # Assuming board is attached to end-effector
        cam2board = base2cam @ base2ee @ ee2board
        
        # Extract rotation and translation
        R = cam2board[:3, :3]
        t = cam2board[:3, 3]
        
        # Convert rotation matrix to rodrigues
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        
        # Project board points onto image
        image_points, _ = cv2.projectPoints(board_points, rvec, tvec, camera_matrix, dist_coeffs)
        
        # Draw the projected points
        for point in image_points:
            x, y = point.ravel()
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Save image
        output_file = os.path.join(output_folder, os.path.basename(image_file))
        cv2.imwrite(output_file, image)
        
        print(f"Saved verification image: {output_file}")

if __name__ == "__main__":
    main()