import cv2
import numpy as np

# Define the ChArUco board parameters
squaresX = 4
squaresY = 3
squareLength = 0.06  # in meters
markerLength = 0.04  # in meters

# Create dictionary - updated API
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# Create the ChArUco board object - updated API
board = cv2.aruco.CharucoBoard(
    size=(squaresX, squaresY),
    squareLength=squareLength,
    markerLength=markerLength,
    dictionary=dictionary
)

# Generate and save the ChArUco board image
board_img = board.generateImage((800, 600))
cv2.imwrite('Camera Calibration/test_charuco_board.png', board_img)

# Show the generated board
cv2.imshow('ChArUco Board', board_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Uncomment this part when you want to detect markers in an image
'''
# Load an image for detection
image = cv2.imread('Camera Calibration/charuco_calibration.png')
if image is None:
    print("Error: Could not load image")
    exit()

cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Detect ArUco markers - updated API
detector = cv2.aruco.ArucoDetector(dictionary)
markerCorners, markerIds, rejectedImgPoints = detector.detectMarkers(image)

# Interpolate ChArUco corners
if markerIds is not None and len(markerIds) > 0:
    charucoDetector = cv2.aruco.CharucoDetector(board)
    charucoCorners, charucoIds = charucoDetector.detectBoard(image, markerCorners, markerIds)
    
    # Draw the detected ChArUco corners
    if charucoCorners is not None and len(charucoCorners) > 0:
        cv2.aruco.drawDetectedCornersCharuco(image, charucoCorners, charucoIds, (255, 0, 0))

# Display the output
cv2.imshow('ChArUco Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
