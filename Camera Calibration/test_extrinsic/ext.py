import cv2
import numpy as np
import glob
import os

def read_transform(file_path):
    """
    Reads a 4x4 transformation matrix from a text file.
    The file should have 4 rows with 4 space-separated numbers each.
    """
    T = np.loadtxt(file_path)
    if T.shape != (4, 4):
        raise ValueError(f"Transform in {file_path} is not 4x4")
    return T

def get_homogeneous_from_pose(rvec, tvec):
    """
    Convert rotation vector and translation vector (from solvePnP)
    to a 4x4 homogeneous transformation matrix.
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def invert_transform(T):
    """
    Invert a homogeneous transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def main():
    # ---------------------------
    # Define ChArUco board parameters
    squaresX = 4
    squaresY = 3
    squareLength = 0.063  # meters
    markerLength = 0.045  # meters

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary)

    # ---------------------------
    # Camera intrinsics (provided)
    camera_matrix = np.array([
        [644.91407239, 0.0, 650.0619773],
        [0.0, 644.91407239, 361.79122397],
        [0.0, 0.0, 1.0]
    ])
    # Note: dist_coeffs is given as a 1D array. Adjust shape if necessary.
    dist_coeffs = np.array([
         25.6177584, -14.9234213, 0.000695934666, 0.00145448927, 135.439185,
         25.7752723, -14.3253110, 137.556902, 0, 0, 0, 0, 0, 0
    ])

    # ---------------------------
    # Load calibration images and corresponding robot transforms
    image_files = sorted(glob.glob(os.path.join("calibration_data", "frame_*.png")))
    
    robot_poses = []  # Will store 4x4 matrices (gripper -> base)
    board_poses = []  # Will store 4x4 matrices (board -> camera)

    for img_file in image_files:
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        transform_file = os.path.join("calibration_data", base_name + ".txt")
        if not os.path.exists(transform_file):
            print(f"Transform file {transform_file} not found, skipping {img_file}")
            continue

        # Read robot pose (gripper-to-base)
        T_gripper2base = read_transform(transform_file)
        robot_poses.append(T_gripper2base)

        # Load and process the image
        img = cv2.imread(img_file)
        if img is None:
            print(f"Failed to load {img_file}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if corners is None or len(corners) == 0:
            print(f"No markers detected in {img_file}")
            continue

        # Interpolate to get Charuco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if charuco_corners is None or len(charuco_corners) < 4:
            print(f"Not enough Charuco corners detected in {img_file}")
            continue

        # Estimate the pose of the ChArUco board relative to the camera
        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs)
        if not ret:
            print(f"Pose estimation failed for {img_file}")
            continue

        T_board2cam = get_homogeneous_from_pose(rvec, tvec)
        board_poses.append(T_board2cam)

    if len(robot_poses) < 2 or len(board_poses) < 2:
        print("Not enough valid calibration pairs found!")
        return

    # ---------------------------
    # Compute relative motions between successive poses
    # For hand-eye calibration we need:
    #   A_i = inv(T_gripper2base_i) * T_gripper2base_{i+1}
    #   B_i = T_board2cam_i * inv(T_board2cam_{i+1})
    A_rotations, A_translations = [], []
    B_rotations, B_translations = [], []

    for i in range(len(robot_poses) - 1):
        # Relative motion for the robot (gripper motion)
        T1 = robot_poses[i]
        T2 = robot_poses[i + 1]
        A = np.dot(invert_transform(T1), T2)
        A_rotations.append(A[:3, :3])
        A_translations.append(A[:3, 3])

        # Relative motion for the board (target motion as seen by the camera)
        T1_b = board_poses[i]
        T2_b = board_poses[i + 1]
        B = np.dot(T1_b, invert_transform(T2_b))
        B_rotations.append(B[:3, :3])
        B_translations.append(B[:3, 3])

    # ---------------------------
    # Perform hand-eye calibration
    # For an eye-to-hand setup (camera fixed relative to base and board mounted on gripper),
    # cv2.calibrateHandEye returns the transformation from camera to gripper.
    ret, t_cam2gripper = cv2.calibrateHandEye(
        A_rotations, A_translations, B_rotations, B_translations, method=cv2.CALIB_HAND_EYE_TSAI
    )
    R_cam2gripper = ret  # Note: the function returns (R_cam2gripper, t_cam2gripper)
    print("Calibration result (transformation from camera to gripper):")
    print("Rotation matrix:")
    print(R_cam2gripper)
    print("Translation vector:")
    print(t_cam2gripper.flatten())

    # ---------------------------
    # (Optional) Compute the transformation from base to camera.
    # Given a robot pose T_gripper2base and the calibration result T_cam2gripper,
    # one can compute:
    #   T_base2cam = T_gripper2base * inv(T_cam2gripper)
    T_cam2gripper_hom = np.eye(4)
    T_cam2gripper_hom[:3, :3] = R_cam2gripper
    T_cam2gripper_hom[:3, 3] = t_cam2gripper.flatten()
    T_gripper2cam = invert_transform(T_cam2gripper_hom)

    # For example, compute T_base2cam for the first measurement:
    T_base2cam = np.dot(robot_poses[0], T_gripper2cam)
    print("\nExample transformation from base to camera (using first robot pose):")
    print(T_base2cam)

if __name__ == '__main__':
    main()
