# --- START OF FILE segmentation_and_pose_estimation.py ---

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from lang_sam_interfaces.srv import SegmentImage
import threading
import time
import cv2
import random
# Use transforms3d or scipy for quaternion operations if available and preferred,
# otherwise stick with tf_transformations or numpy implementations.
from tf_transformations import quaternion_from_matrix, quaternion_multiply, quaternion_about_axis, euler_from_quaternion, translation_matrix, quaternion_matrix, inverse_matrix


class IntegratedSegmenter(Node):
    def __init__(self, text_prompt="box", save_path=None, display_time=30):
        super().__init__('integrated_segmenter')

        # Initialize service client for LangSAM
        self.client = self.create_client(SegmentImage, 'segment_image')

        # Store parameters
        self.text_prompt = text_prompt
        self.save_path = save_path
        self.display_time = display_time # Time in seconds, 0 waits indefinitely

        # Flags and storage
        self.rgb_image_received = False
        self.depth_image_received = False
        self.latest_rgb_image = None
        self.depth_image = None
        self.processing_started = False # Flag to prevent multiple concurrent processing runs
        self.shutdown_timer = None # Timer object for delayed shutdown

        # Create subscriptions to camera topics
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw', # Verify your topic name
            self.rgb_callback,
            rclpy.qos.qos_profile_sensor_data) # Use sensor data QoS

        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw', # Verify your topic name
            self.depth_callback,
             rclpy.qos.qos_profile_sensor_data) # Use sensor data QoS

        # Camera intrinsics (VERIFY THESE VALUES FOR YOUR CAMERA)
        self.fx = 641.415771484375
        self.fy = 640.7596435546875
        self.cx = 650.3182983398438
        self.cy = 357.72979736328125

        # Define static transform from camera_color_optical_frame to base_link
        # IMPORTANT: VERIFY this static transform matrix for your setup!
        # Example: Realsense D435 mounted on UR arm base
        # This should represent the transform T_base_camera (camera frame expressed in base frame)
        # Run `ros2 run tf2_ros tf2_echo <base_frame> <camera_optical_frame>`
        # Note: tf2_echo provides translation (x,y,z) and rotation (x,y,z,w quaternion)
        # Construct the 4x4 matrix from this.
        # Example placeholder - REPLACE WITH YOUR ACTUAL TRANSFORM
        _translation = [-0.510, 0.010, 1.110] # Example translation
        _rotation_quat = [0.0, 0.0, 0.0, 1.0] # Example rotation (replace!) - quaternion [x,y,z,w]
        # This example needs correction based on tf2_echo. Let's use the one from the original code for now.
        # The matrix provided originally seems plausible if the camera is mounted looking down.
        # Transform that maps points from Camera Frame -> Base Frame
        self.T_base_camera = np.array([
            [0.995, 0.095, 0.007, -0.510],
            [0.095, -0.995, -0.005, 0.010],
            [0.006, 0.006, -1.000, 1.110], # Z pointing down from base view seems plausible
            [0.000, 0.000, 0.000, 1.000]
        ])
        # Pre-calculate the inverse transform (Base Frame -> Camera Frame)
        try:
            self.T_camera_base = inverse_matrix(self.T_base_camera)
        except np.linalg.LinAlgError:
            self.get_logger().fatal("Static camera-to-base transform matrix is singular. Cannot proceed.")
            # Exit or raise an exception appropriately
            rclpy.shutdown()
            sys.exit(1)


        # Configure matplotlib
        plt.switch_backend('agg') # Use non-interactive backend suitable for saving files

        self.get_logger().info(f'Integrated Segmenter initialized with prompt: "{text_prompt}"')
        self.get_logger().info('Waiting for RGB and depth images...')


    # --- Quaternion and Transform Helper Functions ---
    def get_rotation_matrix_from_quaternion(self, q):
        """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
        # Uses tf_transformations implementation implicitly via quaternion_matrix
        M = quaternion_matrix(q)
        return M[:3, :3]

    def get_transform_matrix(self, position, quaternion):
        """Creates a 4x4 transformation matrix from position and quaternion."""
        T = translation_matrix(position)
        R = quaternion_matrix(quaternion)
        return T @ R # Matrix multiplication


    # --- Callbacks and Service Handling ---
    def rgb_callback(self, msg):
        """Callback for RGB image messages."""
        if self.processing_started: return
        try:
            # Simplified image conversion assuming common formats or cv_bridge
            if msg.encoding == 'rgb8':
                image_data = np.frombuffer(msg.data, dtype=np.uint8)
                self.latest_rgb_image = image_data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'bgr8':
                image_data = np.frombuffer(msg.data, dtype=np.uint8)
                bgr_image = image_data.reshape((msg.height, msg.width, 3))
                self.latest_rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            else:
                 # Attempt conversion using cv_bridge if available
                 try:
                      import cv_bridge
                      bridge = cv_bridge.CvBridge()
                      self.latest_rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                 except ImportError:
                      self.get_logger().error(f"Unsupported RGB encoding '{msg.encoding}' and cv_bridge not found.")
                      self.latest_rgb_image = None # Ensure it's invalidated
                      return
                 except Exception as e_bridge:
                      self.get_logger().error(f"cv_bridge failed to convert '{msg.encoding}' to rgb8: {e_bridge}")
                      self.latest_rgb_image = None
                      return

            self.rgb_image_received = True
            if self.rgb_image_received and self.depth_image_received and not self.processing_started:
                self.processing_started = True
                self.get_logger().debug("RGB and Depth images received. Starting processing.")
                self.process_images()

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')
            import traceback; self.get_logger().error(traceback.format_exc())
            self.processing_started = False

    def depth_callback(self, msg):
        """Callback for Depth image messages."""
        if self.processing_started: return
        try:
            if msg.encoding != '16UC1':
                self.get_logger().error(f'Unsupported depth encoding: {msg.encoding}. Expected 16UC1.')
                self.depth_image = None # Invalidate
                return

            depth_data = np.frombuffer(msg.data, dtype=np.uint16)
            self.depth_image = depth_data.reshape((msg.height, msg.width))
            self.depth_image_received = True
            # Processing triggered by rgb_callback after both are received

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')
            import traceback; self.get_logger().error(traceback.format_exc())
            self.processing_started = False


    def process_images(self):
        """Initiates the image processing pipeline."""
        local_rgb_image = self.latest_rgb_image
        local_depth_image = self.depth_image

        if local_rgb_image is None or local_depth_image is None:
            self.get_logger().warn("RGB or Depth image is None when starting processing. Aborting.")
            self.processing_started = False
            return

        self.get_logger().info('Processing images...')

        rgb_msg = Image()
        rgb_msg.header.stamp = self.get_clock().now().to_msg()
        rgb_msg.header.frame_id = "camera_color_optical_frame" # Verify this frame ID
        rgb_msg.height = local_rgb_image.shape[0]
        rgb_msg.width = local_rgb_image.shape[1]
        rgb_msg.encoding = "rgb8"
        rgb_msg.is_bigendian = False
        rgb_msg.step = local_rgb_image.shape[1] * 3
        rgb_msg.data = local_rgb_image.tobytes()

        self.send_request(rgb_msg, self.text_prompt)


    def send_request(self, ros_img, text_prompt):
        """Sends image and prompt to the LangSAM service."""
        request = SegmentImage.Request()
        request.image = ros_img
        request.text_prompt = text_prompt

        if not self.client.service_is_ready():
             self.get_logger().warn('LangSAM service not ready yet. Waiting...')
             if not self.client.wait_for_service(timeout_sec=5.0):
                 self.get_logger().error('LangSAM Service not available after timeout')
                 self.processing_started = False
                 return
             self.get_logger().info('LangSAM service is now ready.')

        self.get_logger().info(f'Sending request to LangSAM service for prompt: "{text_prompt}"')
        future = self.client.call_async(request)
        future.add_done_callback(self.process_response_callback)


    def process_response_callback(self, future):
        """Handles the response from the LangSAM service."""
        try:
            response = future.result()
            if not response:
                 self.get_logger().error("Failed to get response from LangSAM service (response is None).")
                 self.processing_started = False
                 return

            self.get_logger().info(f'LangSAM found {len(response.labels)} objects matching "{self.text_prompt}".')

            # Use the images captured *before* the service call for consistency
            current_rgb_image = self.latest_rgb_image.copy() if self.latest_rgb_image is not None else None
            current_depth_image = self.depth_image.copy() if self.depth_image is not None else None

            if current_rgb_image is None or current_depth_image is None:
                self.get_logger().error('Images became unavailable during LangSAM processing.')
                self.processing_started = False
                return

            visualized_image = current_rgb_image.copy()
            pose_info_by_object = {} # Store final pose info for objects with valid faces
            original_pixels_by_object = {}

            # --- Process Each Detected Object Mask ---
            for i in range(len(response.labels)):
                label = response.labels[i]
                score = response.scores[i]
                # img_centroid = response.centroids[i] # 2D centroid from LangSAM

                self.get_logger().info(f'Processing Object {i}: {label} (score: {score:.2f})')

                mask_data = np.frombuffer(response.mask_images[i].data, dtype=np.uint8)
                mask = mask_data.reshape((response.mask_images[i].height, response.mask_images[i].width))

                # --- Extract 3D points (in base_link frame) AND their original pixel coordinates ---
                result = self.get_points_from_mask(mask, current_depth_image)

                points_3d_base = None
                original_pixels_xy = None
                if result is not None:
                    points_3d_base, original_pixels_xy = result
                    if original_pixels_xy is not None:
                         original_pixels_by_object[i] = original_pixels_xy
                else:
                    self.get_logger().warn(f"Could not get 3D points for object {i}.")

                # --- NEW: Find Top Face and Calculate Aligned Pose ---
                final_pose_info = None
                if points_3d_base is not None and len(points_3d_base) > 50: # Check sufficient points
                    self.get_logger().debug(f"Attempting pose calculation for object {i} with {len(points_3d_base)} points.")
                    final_pose_info = self.calculate_aligned_top_face_pose(points_3d_base)

                    if final_pose_info:
                        # Add the original pixel coordinates corresponding to the RANSAC inliers
                        inlier_indices = final_pose_info.get("inlier_indices") # Indices relative to points_3d_base
                        if inlier_indices is not None and original_pixels_xy is not None:
                            try:
                                 final_pose_info["inlier_pixels_xy"] = original_pixels_xy[inlier_indices]
                            except IndexError:
                                 self.get_logger().error(f"IndexError getting inlier pixels for object {i}. Indices: {len(inlier_indices)}, Pixels: {len(original_pixels_xy)}")
                                 final_pose_info["inlier_pixels_xy"] = None
                        else:
                             final_pose_info["inlier_pixels_xy"] = None

                        pose_info_by_object[i] = final_pose_info # Store if valid pose found
                        self.get_logger().info(f'-> Found top face and calculated aligned pose for object {i} (Z: {final_pose_info["position"][2]:.3f}m)')
                    else:
                        self.get_logger().info(f"-> Could not find a suitable top face or calculate pose for object {i}.")
                else:
                     self.get_logger().info(f"-> Not enough valid 3D points ({len(points_3d_base) if points_3d_base is not None else 0}) for object {i} pose calculation.")

                # --- Visualization (Mask Overlay) ---
                color = plt.cm.viridis(score)[:3]
                color = (np.array(color) * 255).astype(np.uint8)
                colored_mask = np.zeros_like(visualized_image)
                mask_bool = mask > 0
                for c in range(3):
                    colored_mask[:, :, c] = np.where(mask_bool, color[c], 0)
                alpha = 0.3
                visualized_image[mask_bool] = (visualized_image[mask_bool] * (1 - alpha) + colored_mask[mask_bool] * alpha).astype(np.uint8)


            # --- Find Tallest Object Among Those with Valid Poses ---
            tallest_object_id = None
            max_height = float('-inf')
            tallest_pose_info = None

            for obj_id, pose_info in pose_info_by_object.items():
                if pose_info and "position" in pose_info:
                    height = pose_info["position"][2]
                    if height > max_height:
                        max_height = height
                        tallest_object_id = obj_id
                        tallest_pose_info = pose_info

            # --- Visualization of Tallest Object Details ---
            if tallest_object_id is not None and tallest_pose_info is not None:
                self.get_logger().info(f"Object {tallest_object_id} selected as tallest with top face at Z={max_height:.3f}m.")

                position = tallest_pose_info["position"]
                final_gripper_quaternion = tallest_pose_info["orientation_quat"]
                wrist_angle = tallest_pose_info["wrist_angle"]

                self.get_logger().info(f'--- Final Gripper Pose for Tallest Object {tallest_object_id} (PCA Aligned) ---')
                self.get_logger().info(f'Position (base): [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]')
                self.get_logger().info(f'Orientation (quat): [{final_gripper_quaternion[0]:.4f}, {final_gripper_quaternion[1]:.4f}, {final_gripper_quaternion[2]:.4f}, {final_gripper_quaternion[3]:.4f}]')
                self.get_logger().info(f'Wrist Angle (rad): {wrist_angle:.4f} ({np.degrees(wrist_angle):.2f} deg)')

                # Optional: Log edge positions if corners are calculated
                # self.calculate_edge_positions_in_gripper_frame(tallest_pose_info) # Needs adaptation if corners logic changes

                # --- Highlight Inlier Pixels for Tallest Face ---
                inlier_pixels = tallest_pose_info.get("inlier_pixels_xy")
                if inlier_pixels is not None and len(inlier_pixels) > 0:
                    self.get_logger().info(f"Highlighting {len(inlier_pixels)} RANSAC inlier pixels for the tallest face.")
                    inlier_pixels = np.round(inlier_pixels).astype(int)
                    img_h, img_w = visualized_image.shape[:2]
                    valid_idx = (inlier_pixels[:, 0] >= 0) & (inlier_pixels[:, 0] < img_w) & \
                                (inlier_pixels[:, 1] >= 0) & (inlier_pixels[:, 1] < img_h)
                    inlier_pixels = inlier_pixels[valid_idx]

                    highlight_color_bgr = (255, 0, 255) # Magenta in BGR
                    temp_vis_img_bgr = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR)
                    for px, py in inlier_pixels:
                         cv2.circle(temp_vis_img_bgr, (px, py), radius=1, color=highlight_color_bgr, thickness=-1)
                    visualized_image = cv2.cvtColor(temp_vis_img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    self.get_logger().warn("No inlier pixel coordinates available for highlighting.")

                # Add marker for tallest object (using the 3D centroid projection)
                center_2d = self.project_point_to_image(position)
                if center_2d is not None:
                    cx, cy = center_2d
                    img_h, img_w = visualized_image.shape[:2]
                    if 0 <= cx < img_w and 0 <= cy < img_h:
                        cv2.drawMarker(visualized_image, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_STAR,
                                    markerSize=30, thickness=3)
                        cv2.putText(visualized_image, "TALLEST", (cx+15, cy-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Visualize the final aligned gripper axes
                visualized_image = self.visualize_final_gripper_pose(visualized_image, tallest_pose_info)

            else:
                self.get_logger().warn('No objects with a valid aligned top face pose were found.')

            # --- Save and Display ---
            save_path = self.save_path or 'segmentation_result_pca_inliers.png'
            try:
                plt.figure(figsize=(15, 10))
                plt.imshow(visualized_image)
                plt.axis('off')
                plt.title(f"Segmentation: '{self.text_prompt}'. Tallest Top Face Inliers Highlighted.")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
                self.get_logger().info(f'Saved result with highlighted inliers to {save_path}')
            except Exception as e_save:
                 self.get_logger().error(f"Failed to save image using matplotlib: {e_save}")
                 try: # Fallback to OpenCV
                      cv2.imwrite(save_path, cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
                      self.get_logger().info(f'Saved result with highlighted inliers to {save_path} using OpenCV.')
                 except Exception as e_cv_save:
                      self.get_logger().error(f"Failed to save image using OpenCV as well: {e_cv_save}")

            # --- OpenCV Display Window ---
            try:
                 cv_display_image = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR)
                 max_disp_w, max_disp_h = 1280, 720
                 h, w = cv_display_image.shape[:2]
                 if w > max_disp_w or h > max_disp_h:
                      scale = min(max_disp_w/w, max_disp_h/h)
                      nw, nh = int(w*scale), int(h*scale)
                      cv_display_image = cv2.resize(cv_display_image, (nw, nh), interpolation=cv2.INTER_AREA)
                 cv2.imshow('Result PCA Aligned + Inliers', cv_display_image)
                 cv2.waitKey(0.2)
            except Exception as e_cv:
                 self.get_logger().warn(f"Could not display image with OpenCV: {e_cv}")

            # --- Shutdown Logic ---
            self.get_logger().info("Processing finished for this frame.")
            if self.display_time > 0:
                self.get_logger().info(f"Waiting {self.display_time} seconds before shutdown...")
                if self.shutdown_timer is None or not self.shutdown_timer.is_alive():
                     self.shutdown_timer = threading.Timer(self.display_time, self.shutdown_node)
                     self.shutdown_timer.daemon = True
                     self.shutdown_timer.start()
            elif self.display_time == 0:
                self.get_logger().info("Display time is 0. Node will continue running. Press Ctrl+C to exit.")
            else: # display_time < 0
                 self.get_logger().info("Display time is negative. Shutting down immediately.")
                 threading.Timer(0.1, self.shutdown_node).start()

        except Exception as e:
            self.get_logger().error(f'Critical error in process_response_callback: {e}')
            import traceback; self.get_logger().error(traceback.format_exc())
            threading.Timer(0.1, self.shutdown_node).start()
        finally:
            self.processing_started = False
            self.get_logger().debug("Processing flag reset.")


    # --- Point Cloud and Coordinate Transforms ---

    def get_points_from_mask(self, mask, depth_image_in):
        """
        Extract 3D points (in robot base frame) AND original (x, y) pixel coords.
        Returns: Tuple (points_3d_robot, original_pixels_xy) or None.
        """
        if depth_image_in is None or mask is None:
            self.get_logger().warn("get_points_from_mask: Missing depth image or mask.")
            return None

        depth_height, depth_width = depth_image_in.shape
        mask_height, mask_width = mask.shape
        mask_resized = mask
        if depth_height != mask_height or depth_width != mask_width:
            self.get_logger().warn(f"Resizing mask ({mask_width}x{mask_height}) to match depth image ({depth_width}x{depth_height})")
            try:
                mask_resized = cv2.resize(mask, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
            except Exception as e_resize:
                 self.get_logger().error(f"Failed to resize mask: {e_resize}"); return None

        y_indices, x_indices = np.nonzero(mask_resized)
        if len(y_indices) == 0: return None

        depth_values_mm = depth_image_in[y_indices, x_indices]
        min_depth_mm, max_depth_mm = 100, 8000
        valid_depth_indices = (depth_values_mm > min_depth_mm) & (depth_values_mm < max_depth_mm)

        valid_x = x_indices[valid_depth_indices]
        valid_y = y_indices[valid_depth_indices]
        valid_depths_mm = depth_values_mm[valid_depth_indices]
        if len(valid_depths_mm) == 0: return None

        original_pixels_xy = np.column_stack((valid_x, valid_y))

        # --- Calculate 3D points in camera frame ---
        depths_m = valid_depths_mm / 1000.0
        X_cam = (valid_x - self.cx) * depths_m / self.fx
        Y_cam = (valid_y - self.cy) * depths_m / self.fy
        Z_cam = depths_m
        points_3d_camera = np.column_stack((X_cam, Y_cam, Z_cam))

        # --- Transform to robot base frame ---
        points_3d_robot = self.transform_points_camera_to_base(points_3d_camera)
        if points_3d_robot is None or len(points_3d_robot) == 0: return None

        self.get_logger().debug(f"Extracted {len(points_3d_robot)} valid 3D points in base frame.")
        return points_3d_robot, original_pixels_xy

    def transform_points_camera_to_base(self, points_3d_camera):
        """Transform points from camera optical frame to robot base frame using pre-defined self.T_base_camera."""
        if points_3d_camera is None or len(points_3d_camera) == 0: return None
        points_3d_camera = np.asarray(points_3d_camera)
        if points_3d_camera.ndim == 1: points_3d_camera = points_3d_camera.reshape(1,-1)

        n_points = points_3d_camera.shape[0]
        points_homogeneous = np.hstack((points_3d_camera, np.ones((n_points, 1))))
        # Apply transform: P_base = T_base_camera * P_camera
        transformed_points_homogeneous = (self.T_base_camera @ points_homogeneous.T).T
        return transformed_points_homogeneous[:, :3]

    def transform_points_base_to_camera(self, points_3d_robot):
        """Transform points from robot base frame back to camera optical frame using pre-defined self.T_camera_base."""
        if points_3d_robot is None or len(points_3d_robot) == 0: return None
        points_3d_robot = np.asarray(points_3d_robot)
        if points_3d_robot.ndim == 1: points_3d_robot = points_3d_robot.reshape(1,-1)

        n_points = points_3d_robot.shape[0]
        points_homogeneous = np.hstack((points_3d_robot, np.ones((n_points, 1))))
        # Apply inverse transform: P_camera = T_camera_base * P_base
        transformed_points_homogeneous = (self.T_camera_base @ points_homogeneous.T).T
        return transformed_points_homogeneous[:, :3]

    def transform_points_base_to_frame(self, points_base, frame_position, frame_quaternion):
        """Transforms points from robot base frame to a target frame defined by its pose in the base frame."""
        if points_base is None or len(points_base) == 0:
            return np.array([])
        points_base = np.asarray(points_base)
        if points_base.ndim == 1:
            points_base = points_base.reshape(1, -1)

        # Transformation matrix from target frame TO base frame
        T_base_frame = self.get_transform_matrix(frame_position, frame_quaternion)

        # Inverse transformation matrix from base frame TO target frame
        try:
            T_frame_base = inverse_matrix(T_base_frame)
        except np.linalg.LinAlgError:
            self.get_logger().error("Target frame transformation matrix is singular, cannot invert.")
            return None

        # Transform points
        n_points = points_base.shape[0]
        points_base_hom = np.hstack((points_base, np.ones((n_points, 1))))
        # P_frame = T_frame_base * P_base
        points_frame_hom = (T_frame_base @ points_base_hom.T).T
        return points_frame_hom[:, :3]


    # --- PCA and Plane Fitting ---

    def perform_2d_pca(self, points_2d):
        """Performs PCA on a set of 2D points using only NumPy.
           Returns: mean, principal_axis_1, principal_axis_2, eigenvalues
        """
        if points_2d is None or len(points_2d) < 2:
            self.get_logger().warn("PCA requires at least 2 points.")
            return None, None, None, None
        points_2d = np.asarray(points_2d)
        if points_2d.shape[1] != 2:
             self.get_logger().error(f"perform_2d_pca expects Nx2 array, got {points_2d.shape}")
             return None, None, None, None

        mean = np.mean(points_2d, axis=0)
        centered_points = points_2d - mean
        if centered_points.shape[0] < 2:
             self.get_logger().warn(f"PCA needs > 1 point for covariance, got {centered_points.shape[0]}.")
             # Return default axes if only one unique point effectively
             return mean, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0]) # Eigenvalues ~0

        # rowvar=False: rows are observations (points), columns are variables (x, y)
        cov_matrix = np.cov(centered_points, rowvar=False)
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            self.get_logger().error("Covariance matrix contains NaN/Inf.")
            # Return default axes
            return mean, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])

        try: # Use eigh for symmetric matrices
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        except np.linalg.LinAlgError:
             self.get_logger().error("PCA failed: Eigenvalue decomposition error.")
             return mean, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])

        # Sort eigenvalues and eigenvectors in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices] # Columns are eigenvectors

        principal_axis_1 = eigenvectors[:, 0]
        principal_axis_2 = eigenvectors[:, 1]
        self.get_logger().debug(f"PCA 2D: Axis1={np.round(principal_axis_1,3)}, Axis2={np.round(principal_axis_2,3)}, Values={np.round(eigenvalues,4)}")
        return mean, principal_axis_1, principal_axis_2, eigenvalues


    def get_plane_info_ransac(self, points_3d, iterations=100, threshold=0.01):
        """
        Calculates plane using RANSAC. Returns dict including inlier_indices relative to input points_3d.
        """
        points_array = np.asarray(points_3d)
        num_points = len(points_array)
        if num_points < 3: return None

        best_inliers_count = -1
        best_plane_params = None
        best_inlier_indices_relative = None

        for _ in range(iterations):
            try:
                sample_indices = random.sample(range(num_points), 3)
                sample_points = points_array[sample_indices]
            except ValueError: continue

            p1, p2, p3 = sample_points[0], sample_points[1], sample_points[2]
            v1 = p2 - p1; v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-6: continue

            normal = normal / normal_norm
            d = -np.dot(normal, p1)
            distances = np.abs(points_array @ normal + d)
            current_inlier_indices = np.where(distances < threshold)[0]
            current_inliers_count = len(current_inlier_indices)

            if current_inliers_count > best_inliers_count:
                best_inliers_count = current_inliers_count
                best_plane_params = (normal, d)
                best_inlier_indices_relative = current_inlier_indices
                # Optional early exit heuristic
                if best_inliers_count > num_points * 0.8: break

        min_ransac_inliers = 25 # Minimum number of inliers required
        if best_plane_params is None or best_inliers_count < min_ransac_inliers:
            self.get_logger().debug(f"RANSAC failed: best plane had {best_inliers_count} inliers (< {min_ransac_inliers}).")
            return None

        final_normal, _ = best_plane_params
        final_inlier_points = points_array[best_inlier_indices_relative]
        center = np.mean(final_inlier_points, axis=0)

        result = {
            "normal": final_normal, # Normal vector of the plane
            "center": center,      # Centroid of the inlier points
            "inliers": final_inlier_points, # 3D coordinates of inliers
            "inlier_indices": best_inlier_indices_relative, # Indices relative to input points_3d
            "inlier_count": best_inliers_count,
        }
        self.get_logger().debug(f"RANSAC success: Found plane with {best_inliers_count} inliers.")
        return result

    # --- NEW Pose Calculation Function ---
    def calculate_aligned_top_face_pose(self, points_3d_base, ransac_iterations=300, ransac_threshold=0.008):
        """
        Finds the top face via RANSAC, transforms inliers to an initial gripper frame,
        performs 2D PCA in that frame, calculates the wrist alignment angle,
        and returns the final gripper pose (position, orientation).

        Args:
            points_3d_base (np.ndarray): Nx3 array of points in the robot base frame.
            ransac_iterations (int): Number of iterations for RANSAC.
            ransac_threshold (float): Inlier distance threshold for RANSAC.

        Returns:
            dict: Dictionary containing final pose info ('position', 'orientation_quat',
                  'wrist_angle', 'normal', 'inliers', 'inlier_indices', etc.)
                  or None if unsuccessful.
        """
        if points_3d_base is None or len(points_3d_base) < 3:
            self.get_logger().warn("calculate_aligned_top_face_pose: Not enough points provided.")
            return None

        # --- 1. Find Best Candidate Top Plane using RANSAC ---
        plane_info = self.get_plane_info_ransac(points_3d_base, ransac_iterations, ransac_threshold)

        if not plane_info:
            self.get_logger().warn("RANSAC did not find a suitable plane.")
            return None

        # Ensure normal points upwards (positive Z in robot base frame)
        normal = plane_info["normal"]
        if normal[2] < 0:
            self.get_logger().debug("RANSAC normal was pointing down, flipping.")
            normal = -normal
            plane_info["normal"] = normal # Update the dictionary too

        # Check if it's sufficiently "top-facing"
        min_up_alignment = 0.70 # Cosine of angle with Z-axis
        if normal[2] < min_up_alignment:
            self.get_logger().info(f"Plane found, but normal {np.round(normal,3)} not pointing up enough (Z={normal[2]:.3f} < {min_up_alignment}). Skipping.")
            return None

        center_base = plane_info["center"]
        inliers_base = plane_info["inliers"]
        inlier_indices = plane_info["inlier_indices"]
        inlier_count = plane_info["inlier_count"]
        self.get_logger().info(f"Top Plane Found: Center Z={center_base[2]:.3f}, Normal={np.round(normal,3)}, Inliers={inlier_count}")

        # --- 2. Define Initial Gripper Frame (at Center, Z opposite normal) ---
        # Gripper Z-axis points opposite to surface normal for a top grasp
        z_axis_gripper = -normal
        z_axis_gripper /= np.linalg.norm(z_axis_gripper) # Normalize

        # Gripper X-axis: Try to align with projected base X-axis onto the plane
        base_x_axis = np.array([1.0, 0.0, 0.0])
        # Project base_x onto the plane: x_proj = base_x - (base_x . normal) * normal
        # Since z_axis_gripper = -normal, normal = -z_axis_gripper
        # x_proj = base_x - (base_x . (-z_axis_gripper)) * (-z_axis_gripper)
        # x_proj = base_x - (base_x . z_axis_gripper) * z_axis_gripper
        initial_x_axis = base_x_axis - np.dot(base_x_axis, z_axis_gripper) * z_axis_gripper

        # Check if initial X is too small (i.e., Z-gripper is aligned with Base X)
        if np.linalg.norm(initial_x_axis) < 1e-5:
            self.get_logger().debug("Gripper Z nearly parallel to Base X, using Base Y for initial X calculation.")
            # Use Base Y axis instead to define the plane orientation
            base_y_axis = np.array([0.0, 1.0, 0.0])
            initial_x_axis = base_y_axis - np.dot(base_y_axis, z_axis_gripper) * z_axis_gripper
            if np.linalg.norm(initial_x_axis) < 1e-5: # Should not happen if normal is not [0,0,1]
                 self.get_logger().warn("Cannot determine initial X axis robustly. Using arbitrary orthogonal axis.")
                 # Create an arbitrary vector not parallel to z_axis_gripper
                 arbitrary_vec = np.array([1.0, 0.0, 0.0]) if abs(z_axis_gripper[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                 initial_y_axis = np.cross(z_axis_gripper, arbitrary_vec)
                 initial_y_axis /= np.linalg.norm(initial_y_axis)
                 initial_x_axis = np.cross(initial_y_axis, z_axis_gripper) # Recalculate X from YxZ
                 initial_x_axis /= np.linalg.norm(initial_x_axis) # Normalize X too
            else:
                 initial_x_axis /= np.linalg.norm(initial_x_axis)
                 # Gripper Y-axis: y = z x x (Right Hand Rule)
                 initial_y_axis = np.cross(z_axis_gripper, initial_x_axis)
                 initial_y_axis /= np.linalg.norm(initial_y_axis) # Normalize Y
        else:
            initial_x_axis /= np.linalg.norm(initial_x_axis)
            # Gripper Y-axis: y = z x x (Right Hand Rule)
            initial_y_axis = np.cross(z_axis_gripper, initial_x_axis)
            initial_y_axis /= np.linalg.norm(initial_y_axis) # Normalize Y


        # Create initial rotation matrix (columns are axes in base frame)
        initial_rot_matrix_base = np.eye(4)
        initial_rot_matrix_base[:3, 0] = initial_x_axis
        initial_rot_matrix_base[:3, 1] = initial_y_axis
        initial_rot_matrix_base[:3, 2] = z_axis_gripper
        initial_gripper_quat_base = quaternion_from_matrix(initial_rot_matrix_base)
        self.get_logger().debug(f"Initial Gripper Quat (Base): {np.round(initial_gripper_quat_base, 3)}")

        # --- 3. Transform Inliers to this Initial Gripper Frame ---
        inliers_in_initial_gripper_frame = self.transform_points_base_to_frame(
            inliers_base, center_base, initial_gripper_quat_base
        )
        if inliers_in_initial_gripper_frame is None or len(inliers_in_initial_gripper_frame) == 0:
             self.get_logger().error("Failed to transform points to initial gripper frame for PCA.")
             return None # Cannot proceed

        # Check Z-values in the initial gripper frame - should be close to zero
        avg_z_in_gripper = np.mean(inliers_in_initial_gripper_frame[:, 2])
        std_z_in_gripper = np.std(inliers_in_initial_gripper_frame[:, 2])
        self.get_logger().debug(f"Inliers in Initial Gripper Frame: Avg Z={avg_z_in_gripper:.4f}, Std Z={std_z_in_gripper:.4f}")
        if abs(avg_z_in_gripper) > 0.01 or std_z_in_gripper > 0.01: # Thresholds might need tuning
            self.get_logger().warn("Inlier points have significant Z-spread in the initial gripper frame - plane fit might be poor or transform incorrect.")

        # --- 4. Perform 2D PCA on XY coordinates in Initial Gripper Frame ---
        points_2d_gripper = inliers_in_initial_gripper_frame[:, :2] # Select X, Y columns
        pca_mean_2d, pca_axis1_2d, pca_axis2_2d, pca_eigenvalues = self.perform_2d_pca(points_2d_gripper)

        if pca_axis1_2d is None:
            self.get_logger().warn("PCA failed on projected points. Cannot determine alignment angle.")
            # Return pose based on initial orientation (no wrist rotation)
            wrist_angle = 0.0
            final_gripper_quat_base = initial_gripper_quat_base
        else:
            # --- 5. Calculate Wrist Angle ---
            # Angle needed to rotate the initial gripper's X-axis ([1, 0])
            # to align with the primary PCA component (pca_axis1_2d)
            # This rotation happens around the initial gripper's Z-axis.
            wrist_angle = np.arctan2(pca_axis1_2d[1], pca_axis1_2d[0])
            self.get_logger().info(f"Calculated PCA alignment wrist angle: {np.degrees(wrist_angle):.2f} degrees")

            # --- 6. Calculate Final Gripper Orientation ---
            # Create quaternion for rotation around the *local* Z-axis ([0, 0, 1])
            rotation_around_local_z = quaternion_about_axis(wrist_angle, [0, 0, 1])

            # Apply this local rotation to the initial orientation
            # Final Orientation = Initial Orientation * Local Rotation
            final_gripper_quat_base = quaternion_multiply(initial_gripper_quat_base, rotation_around_local_z)
            final_gripper_quat_base /= np.linalg.norm(final_gripper_quat_base) # Normalize

        self.get_logger().debug(f"Final Gripper Quat (Base): {np.round(final_gripper_quat_base, 3)}")

        # --- 7. Assemble Result ---
        result_info = {
            "position": center_base, # Final position is the centroid in base frame
            "orientation_quat": final_gripper_quat_base, # Final orientation in base frame
            "wrist_angle": wrist_angle, # Rotation applied around local Z
            "normal": normal, # Plane normal in base frame
            "inliers": inliers_base, # RANSAC inliers in base frame
            "inlier_indices": inlier_indices, # Indices relative to original points_3d_base input
            "inlier_count": inlier_count,
            "initial_gripper_quaternion": initial_gripper_quat_base, # For debugging/comparison
            "pca_axis1_2d_gripper": pca_axis1_2d, # PCA axis in initial gripper's XY plane
            "pca_axis2_2d_gripper": pca_axis2_2d, # PCA axis in initial gripper's XY plane
            "pca_eigenvalues": pca_eigenvalues,
        }
        return result_info


    # --- Visualization and Utility functions ---

    def visualize_final_gripper_pose(self, visualized_image, pose_info):
        """Visualizes the FINAL aligned gripper axes."""
        center_3d_base = pose_info["position"]
        final_quat_base = pose_info["orientation_quat"]

        center_2d = self.project_point_to_image(center_3d_base)
        if center_2d is None:
            self.get_logger().warn("Cannot visualize alignment, center point projects outside image.")
            return visualized_image

        vis_img_bgr = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR) # Convert for drawing

        # Get final axes directions in base frame
        rot_matrix_base = self.get_rotation_matrix_from_quaternion(final_quat_base)
        x_axis_base = rot_matrix_base[:, 0]
        y_axis_base = rot_matrix_base[:, 1]
        z_axis_base = rot_matrix_base[:, 2] # Should point roughly downwards

        axis_length_m = 0.07 # Length of axes lines in meters

        # Calculate 3D endpoints of axes in base frame
        x_end_3d_base = center_3d_base + x_axis_base * axis_length_m
        y_end_3d_base = center_3d_base + y_axis_base * axis_length_m
        z_end_3d_base = center_3d_base + z_axis_base * axis_length_m

        # Project 3D endpoints to 2D image coordinates
        x_end_2d = self.project_point_to_image(x_end_3d_base)
        y_end_2d = self.project_point_to_image(y_end_3d_base)
        z_end_2d = self.project_point_to_image(z_end_3d_base)

        # Draw axes lines
        axis_color_x_bgr = (0, 0, 255) # Red (X)
        axis_color_y_bgr = (0, 255, 0) # Green (Y)
        axis_color_z_bgr = (255, 0, 0) # Blue (Z)
        axis_thickness = 2

        if x_end_2d: cv2.line(vis_img_bgr, center_2d, x_end_2d, axis_color_x_bgr, axis_thickness)
        if y_end_2d: cv2.line(vis_img_bgr, center_2d, y_end_2d, axis_color_y_bgr, axis_thickness)
        if z_end_2d: cv2.line(vis_img_bgr, center_2d, z_end_2d, axis_color_z_bgr, axis_thickness)

        # Add labels near end points
        font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; font_thickness = 1; label_offset = (5, 5)
        if x_end_2d: cv2.putText(vis_img_bgr, "X", (x_end_2d[0]+label_offset[0], x_end_2d[1]+label_offset[1]), font, font_scale, axis_color_x_bgr, font_thickness+1, cv2.LINE_AA)
        if y_end_2d: cv2.putText(vis_img_bgr, "Y", (y_end_2d[0]+label_offset[0], y_end_2d[1]+label_offset[1]), font, font_scale, axis_color_y_bgr, font_thickness+1, cv2.LINE_AA)
        # if z_end_2d: cv2.putText(vis_img_bgr, "Z", (z_end_2d[0]+label_offset[0], z_end_2d[1]+label_offset[1]), font, font_scale, axis_color_z_bgr, font_thickness+1, cv2.LINE_AA) # Z label might be cluttered

        # Add Wrist Angle text
        wrist_angle_deg = np.degrees(pose_info["wrist_angle"])
        angle_text = f"Align: {wrist_angle_deg:.1f}d"
        text_pos = (center_2d[0] - 50, center_2d[1] + 40) # Adjust position as needed
        (text_w, text_h), _ = cv2.getTextSize(angle_text, font, font_scale, font_thickness)
        cv2.rectangle(vis_img_bgr, (text_pos[0]-2, text_pos[1]+2), (text_pos[0]+text_w+2, text_pos[1]-text_h-2), (0,0,0), -1) # Black background
        cv2.putText(vis_img_bgr, angle_text, text_pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # White text

        # Convert back to RGB
        visualized_image_out = cv2.cvtColor(vis_img_bgr, cv2.COLOR_BGR2RGB)
        return visualized_image_out

    def project_point_to_image(self, point_3d_robot):
        """Projects a 3D point in robot base frame to 2D image coordinates."""
        if point_3d_robot is None: return None

        # Transform point from robot base frame to camera frame
        point_camera_frame = self.transform_points_base_to_camera(np.array([point_3d_robot]))
        if point_camera_frame is None or len(point_camera_frame) == 0: return None
        point_camera = point_camera_frame[0]

        # Check if point is valid (in front of camera)
        if point_camera[2] <= 0.01: return None # Z must be positive

        # Apply perspective projection
        u = self.fx * point_camera[0] / point_camera[2] + self.cx
        v = self.fy * point_camera[1] / point_camera[2] + self.cy
        u_int, v_int = int(round(u)), int(round(v))

        # Check image bounds
        img_h, img_w = -1, -1
        if self.latest_rgb_image is not None: img_h, img_w = self.latest_rgb_image.shape[:2]
        else: img_h, img_w = 720, 1280 # Default estimate

        if 0 <= u_int < img_w and 0 <= v_int < img_h:
            return (u_int, v_int)
        else:
            return None # Projects outside image


    def shutdown_node(self):
        """Gracefully shuts down the node."""
        self.get_logger().info('Shutdown requested. Cleaning up...')
        if self.shutdown_timer and self.shutdown_timer.is_alive():
            self.shutdown_timer.cancel()
            self.get_logger().info('Shutdown timer cancelled.')
        try: # Close OpenCV window
            if cv2.getWindowProperty('Result PCA Aligned + Inliers', cv2.WND_PROP_VISIBLE) >= 1:
                 cv2.destroyWindow('Result PCA Aligned + Inliers')
                 cv2.waitKey(50)
                 self.get_logger().info('OpenCV window closed.')
        except Exception as e_cv_destroy:
            self.get_logger().debug(f"Error closing OpenCV window: {e_cv_destroy}")
        if rclpy.ok():
            self.get_logger().info('Shutting down ROS context...')
            # self.destroy_node() # Managed by rclpy.shutdown()
            rclpy.shutdown()
        self.get_logger().info('Node shutdown sequence complete.')


# --- Main Execution ---
def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Integrated Segmenter using LangSAM and PCA Alignment')
    parser.add_argument('--prompt', type=str, default='box', help='Text prompt for LangSAM segmentation')
    parser.add_argument('--save', type=str, help='Path to save the result image (e.g., output.png)')
    parser.add_argument('--wait', type=int, default=30, help='Time (seconds) to display results before shutting down. 0 waits indefinitely.')
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])

    segmenter_node = None
    try:
        segmenter_node = IntegratedSegmenter(
            text_prompt=parsed_args.prompt,
            save_path=parsed_args.save,
            display_time=parsed_args.wait
        )
        rclpy.spin(segmenter_node)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt received, shutting down...')
    except Exception as e:
         logger = rclpy.logging.get_logger("main_exception_logger")
         logger.fatal(f"Unhandled exception during node execution: {e}")
         import traceback
         logger.error(traceback.format_exc())
    finally:
        print('Performing final cleanup...')
        if segmenter_node is not None:
             segmenter_node.shutdown_node()
        else:
             print("Node object not available for controlled shutdown. Attempting basic ROS shutdown.")
             if rclpy.ok(): rclpy.shutdown()
        print("Cleanup finished.")

if __name__ == '__main__':
    main()

# --- END OF FILE segmentation_and_pose_estimation.py ---