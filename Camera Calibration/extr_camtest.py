import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation

def load_transform(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    translation = np.array([float(x) for x in lines[0].split()[1:]])
    rotation = np.array([float(x) for x in lines[1].split()[1:]])
    return translation, rotation

def load_image(file_path):
    return cv2.imread(file_path)

def create_charuco_board():
    squaresX, squaresY = 4, 3
    squareLength, markerLength = 0.063, 0.045  # in meters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard(
        size=(squaresX, squaresY),
        squareLength=squareLength,
        markerLength=markerLength,
        dictionary=dictionary
    )
    return board

def detect_charuco(image, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charucoCorners, charucoIds, markerCorners, markerIds = detector.detectBoard(gray)
    return charucoCorners, charucoIds

def calibrate_camera(object_points, image_points, image_size):
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return camera_matrix, dist_coeffs

def main():
    calibration_folder = "calibration_data"
    board = create_charuco_board()

    object_points = []
    image_points = []

    for i in range(1, 101):  # Assuming 100 calibration images
        transform_file = f"transform_{i:03d}.txt"
        image_file = f"frame_{i:03d}.png"
        
        transform_path = os.path.join(calibration_folder, transform_file)
        image_path = os.path.join(calibration_folder, image_file)
        
        if not (os.path.exists(transform_path) and os.path.exists(image_path)):
            continue

        translation, rotation = load_transform(transform_path)
        image = load_image(image_path)
        
        charucoCorners, charucoIds = detect_charuco(image, board)
        
        if charucoCorners is not None and len(charucoCorners) > 0:
            object_points.append(board.getChessboardCorners()[charucoIds].reshape(-1, 3))
            image_points.append(charucoCorners.reshape(-1, 2))

    if not object_points:
        print("No valid calibration data found.")
        return

    image_size = image.shape[:2]
    camera_matrix, dist_coeffs = calibrate_camera(object_points, image_points, image_size)

    print("Camera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)

    # Calculate the camera pose relative to the robot
    for i, (obj_pts, img_pts) in enumerate(zip(object_points, image_points)):
        _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
        
        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)
        
        # Combine rotation and translation into a 4x4 transformation matrix
        camera_transform = np.eye(4)
        camera_transform[:3, :3] = rot_matrix
        camera_transform[:3, 3] = tvec.reshape(3)
        
        # Invert the camera transform to get the camera pose relative to the ChArUco board
        camera_pose = np.linalg.inv(camera_transform)
        
        # Create robot transform from the loaded data
        robot_rotation = Rotation.from_quat(rotation).as_matrix()
        robot_transform = np.eye(4)
        robot_transform[:3, :3] = robot_rotation
        robot_transform[:3, 3] = translation
        
        # Calculate camera pose relative to the robot
        camera_to_robot = np.dot(robot_transform, camera_pose)
        
        print(f"\nCamera pose relative to robot for frame {i+1}:")
        print(camera_to_robot)

if __name__ == "__main__":
    main()
