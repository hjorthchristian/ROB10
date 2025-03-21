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

def main():
    calibration_folder = "calibration_data"
    board = create_charuco_board()

    robot_poses = []
    camera_poses = []

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
            # Compute camera pose
            _, rvec, tvec = cv2.solvePnP(
                board.getChessboardCorners()[charucoIds],
                charucoCorners,
                np.array([[1.12164815e+03, 0.00000000e+00, 5.30183311e+02],[0.00000000e+00, 1.13206527e+03, 5.38532610e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),  # Assume this is pre-computed
                np.array([ 4.99817826e-02, -1.53711017e+00,-1.89962170e-05, -3.19905629e-02,
   5.43504056e+00]) # Assume this is pre-computed
            )
            
            # Convert rotation vector to matrix
            camera_R, _ = cv2.Rodrigues(rvec)
            camera_T = tvec
            
            # Store camera pose
            camera_poses.append((camera_R, camera_T))
            
            # Convert robot pose
            robot_R = Rotation.from_quat(rotation).as_matrix()
            robot_T = translation.reshape(3, 1)
            
            # Store robot pose
            robot_poses.append((robot_R, robot_T))

    # Perform hand-eye calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        robot_poses,
        camera_poses,
        method=cv2.CALIB_HAND_EYE_PARK
    )

    print("Camera to Robot Gripper Rotation:")
    print(R_cam2gripper)
    print("\nCamera to Robot Gripper Translation:")
    print(t_cam2gripper)

if __name__ == "__main__":
    main()
