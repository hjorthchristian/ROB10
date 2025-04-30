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
from tf_transformations import quaternion_from_matrix, quaternion_multiply, quaternion_about_axis, euler_from_quaternion


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

        # Configure matplotlib
        plt.switch_backend('agg') # Use non-interactive backend suitable for saving files

        self.get_logger().info(f'Integrated Segmenter initialized with prompt: "{text_prompt}"')
        self.get_logger().info('Waiting for RGB and depth images...')


    # --- Quaternion Helper Functions ---
    def get_rotation_matrix_from_quaternion(self, q):
        """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
        x, y, z, w = q
        Nq = w*w + x*x + y*y + z*z
        if Nq < 1e-8:
            return np.identity(3)
        s = 2.0 / Nq
        X = x * s; Y = y * s; Z = z * s
        wX = w * X; wY = w * Y; wZ = w * Z
        xX = x * X; xY = x * Y; xZ = x * Z
        yY = y * Y; yZ = y * Z; zZ = z * Z
        # Returns rotation matrix transforming point from G->B (gripper to base)
        # if quaternion represents B->G (base to gripper) orientation
        return np.array([
            [1.0-(yY+zZ), xY-wZ, xZ+wY],
            [xY+wZ, 1.0-(xX+zZ), yZ-wX],
            [xZ-wY, yZ+wX, 1.0-(xX+yY)]
        ])


    # --- Callbacks and Service Handling ---
    def rgb_callback(self, msg):
        """Callback for RGB image messages."""
        # self.get_logger().debug("RGB callback triggered.")
        if self.processing_started: # Avoid processing while previous run is ongoing
            return
        try:
            if msg.encoding != 'rgb8':
                self.get_logger().warn(f'Expected rgb8 encoding, got {msg.encoding}. Attempting conversion.')
                try:
                    import cv_bridge
                    bridge = cv_bridge.CvBridge()
                    self.latest_rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                except ImportError:
                    self.get_logger().error("cv_bridge not found, cannot convert image encoding.")
                    return
                except Exception as e_conv:
                    self.get_logger().error(f"Error converting image with cv_bridge: {e_conv}")
                    # Try fallback for bgr8
                    if msg.encoding == 'bgr8':
                         self.get_logger().warn("Assuming BGR8, converting to RGB.")
                         image_data = np.frombuffer(msg.data, dtype=np.uint8)
                         bgr_image = image_data.reshape((msg.height, msg.width, 3))
                         self.latest_rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    else:
                         self.get_logger().error(f"Cannot process encoding {msg.encoding} without cv_bridge or known fallback.")
                         return
            else:
                 image_data = np.frombuffer(msg.data, dtype=np.uint8)
                 self.latest_rgb_image = image_data.reshape((msg.height, msg.width, 3))

            self.rgb_image_received = True

            # Check if both images are ready and start processing *only if not already started*
            if self.rgb_image_received and self.depth_image_received and not self.processing_started:
                self.processing_started = True # Set flag HERE
                self.get_logger().debug("RGB received, depth already received. Starting processing.")
                # Schedule process_images to run slightly deferred to ensure callbacks complete
                # threading.Timer(0.01, self.process_images).start() # Can help avoid race conditions
                self.process_images() # Or call directly if race conditions aren't an issue


        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.processing_started = False # Reset flag on error

    def depth_callback(self, msg):
        """Callback for Depth image messages."""
        # self.get_logger().debug("Depth callback triggered.")
        if self.processing_started: # Avoid processing while previous run is ongoing
             return
        try:
            if msg.encoding != '16UC1':
                self.get_logger().error(f'Unsupported depth encoding: {msg.encoding}')
                return

            depth_data = np.frombuffer(msg.data, dtype=np.uint16)
            self.depth_image = depth_data.reshape((msg.height, msg.width))
            self.depth_image_received = True
            # No need to trigger processing from here, rgb_callback handles the check

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.processing_started = False # Reset flag on error


    def process_images(self):
        """Initiates the image processing pipeline."""
        # Capture local copies immediately to prevent them changing during processing
        local_rgb_image = self.latest_rgb_image
        local_depth_image = self.depth_image

        if local_rgb_image is None or local_depth_image is None:
            self.get_logger().warn("RGB or Depth image is None when starting processing. Aborting.")
            self.processing_started = False # Reset flag
            return

        self.get_logger().info('Processing images...')

        # Convert NumPy array back to ROS Image msg for service call
        rgb_msg = Image()
        rgb_msg.header.stamp = self.get_clock().now().to_msg()
        # Frame ID should ideally match the camera's published frame
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

        # Check service availability
        if not self.client.service_is_ready():
             self.get_logger().warn('LangSAM service not ready yet. Waiting...')
             if not self.client.wait_for_service(timeout_sec=5.0):
                 self.get_logger().error('LangSAM Service not available after timeout')
                 self.processing_started = False # Allow trying again
                 return
             self.get_logger().info('LangSAM service is now ready.')


        self.get_logger().info(f'Sending request to LangSAM service for prompt: "{text_prompt}"')
        future = self.client.call_async(request)
        # Pass local copies of images to the callback context if needed for absolute certainty
        # future.add_done_callback(lambda f: self.process_response_callback(f, self.latest_rgb_image, self.depth_image))
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
            # Make copies to avoid modification issues if callbacks overlap
            current_rgb_image = self.latest_rgb_image.copy() if self.latest_rgb_image is not None else None
            current_depth_image = self.depth_image.copy() if self.depth_image is not None else None

            if current_rgb_image is None or current_depth_image is None:
                self.get_logger().error('Images became unavailable during LangSAM processing.')
                self.processing_started = False
                return

            visualized_image = current_rgb_image.copy() # Work on a copy for drawing
            plane_info_by_object = {}
            original_pixels_by_object = {} # Store original pixels for each object's mask

            # --- Process Each Detected Object Mask ---
            for i in range(len(response.labels)):
                label = response.labels[i]
                score = response.scores[i]
                img_centroid = response.centroids[i] # 2D centroid from LangSAM

                self.get_logger().info(f'Processing Object {i}: {label} (score: {score:.2f})')

                mask_data = np.frombuffer(response.mask_images[i].data, dtype=np.uint8)
                mask = mask_data.reshape((response.mask_images[i].height, response.mask_images[i].width))

                # --- Extract 3D points AND their original pixel coordinates ---
                result = self.get_points_from_mask(mask, current_depth_image)

                points_3d = None
                original_pixels_xy = None
                if result is not None:
                    points_3d, original_pixels_xy = result
                    if original_pixels_xy is not None:
                         original_pixels_by_object[i] = original_pixels_xy # Store pixels for this object
                else:
                    self.get_logger().warn(f"Could not get 3D points for object {i}.")


                # --- Find Top Face using RANSAC and PCA alignment ---
                top_face_info = None # Initialize for this object
                if points_3d is not None and len(points_3d) > 50: # Check sufficient points
                    top_face_info = self.find_top_face_with_pca_alignment(points_3d)

                    if top_face_info is not None:
                        # Add the original pixel coordinates corresponding to the inliers
                        inlier_indices = top_face_info.get("inlier_indices") # Indices relative to points_3d
                        if inlier_indices is not None and original_pixels_xy is not None:
                            try:
                                 top_face_info["inlier_pixels_xy"] = original_pixels_xy[inlier_indices]
                            except IndexError:
                                 self.get_logger().error(f"IndexError getting inlier pixels for object {i}. Indices: {len(inlier_indices)}, Pixels: {len(original_pixels_xy)}")
                                 top_face_info["inlier_pixels_xy"] = None
                        else:
                             top_face_info["inlier_pixels_xy"] = None

                        plane_info_by_object[i] = top_face_info # Store if valid face found
                        self.get_logger().info(f'-> Found aligned top face for object {i} (Height: {top_face_info["center"][2]:.3f}m)')
                    else:
                        self.get_logger().info(f"-> Could not find a suitable top face plane for object {i}.")
                else:
                     self.get_logger().info(f"-> Not enough valid 3D points ({len(points_3d) if points_3d is not None else 0}) for object {i} plane fitting.")


                # --- Visualization (Mask Overlay) ---
                # Apply mask overlay even if no plane was found
                color = plt.cm.viridis(score)[:3] # Color based on score (viridis map)
                color = (np.array(color) * 255).astype(np.uint8)
                colored_mask = np.zeros_like(visualized_image)
                mask_bool = mask > 0
                for c in range(3):
                    colored_mask[:, :, c] = np.where(mask_bool, color[c], 0)

                alpha = 0.3 # Transparency
                visualized_image[mask_bool] = (visualized_image[mask_bool] * (1 - alpha) + colored_mask[mask_bool] * alpha).astype(np.uint8)


            # --- Find Tallest Object Among Those with Valid Faces ---
            tallest_object_id = None
            max_height = float('-inf')
            tallest_face_info = None

            for obj_id, face_info in plane_info_by_object.items():
                if face_info and "center" in face_info:
                    height = face_info["center"][2]
                    if height > max_height:
                        max_height = height
                        tallest_object_id = obj_id
                        tallest_face_info = face_info

            # --- Visualization of Tallest Object Details ---
            if tallest_object_id is not None and tallest_face_info is not None:
                self.get_logger().info(f"Object {tallest_object_id} selected as tallest with top face at Z={max_height:.3f}m.")

                center = tallest_face_info["center"]
                final_gripper_quaternion = tallest_face_info["final_gripper_quaternion"]
                wrist_angle = tallest_face_info["wrist_angle"]

                self.get_logger().info(f'--- Final Gripper Pose for Tallest Object {tallest_object_id} (PCA Aligned) ---')
                self.get_logger().info(f'Position (base): [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]')
                self.get_logger().info(f'Orientation (quat): [{final_gripper_quaternion[0]:.4f}, {final_gripper_quaternion[1]:.4f}, {final_gripper_quaternion[2]:.4f}, {final_gripper_quaternion[3]:.4f}]')
                self.get_logger().info(f'Wrist_3 Angle (rad): {wrist_angle:.4f} ({np.degrees(wrist_angle):.2f} deg)')

                # Log edge positions relative to the final gripper frame
                self.calculate_edge_positions_in_gripper_frame(tallest_face_info)

                # --- Highlight Inlier Pixels for Tallest Face ---
                inlier_pixels = tallest_face_info.get("inlier_pixels_xy")
                if inlier_pixels is not None and len(inlier_pixels) > 0:
                    self.get_logger().info(f"Highlighting {len(inlier_pixels)} inlier pixels for the tallest face.")
                    inlier_pixels = np.round(inlier_pixels).astype(int)
                    img_h, img_w = visualized_image.shape[:2]
                    valid_idx = (inlier_pixels[:, 0] >= 0) & (inlier_pixels[:, 0] < img_w) & \
                                (inlier_pixels[:, 1] >= 0) & (inlier_pixels[:, 1] < img_h)
                    inlier_pixels = inlier_pixels[valid_idx]
                    
                    # Draw small circles for better visibility than single pixels
                    highlight_color_bgr = (255, 0, 255) # Magenta in BGR for OpenCV
                    temp_vis_img_bgr = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR) # Convert for drawing
                    for px, py in inlier_pixels:
                         cv2.circle(temp_vis_img_bgr, (px, py), radius=1, color=highlight_color_bgr, thickness=-1) # Small filled circles
                    
                    # --- Fit minimum area rectangle to inlier pixels ---
                    rect = cv2.minAreaRect(inlier_pixels)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int32(box_points)
                    
                    # Draw the minimum area rectangle
                    cv2.drawContours(temp_vis_img_bgr, [box_points], 0, (0, 255, 0), 2)  # Green rectangle
                    
                    # Get rectangle center, width, height, and angle
                    rect_center, rect_dims, rect_angle = rect
                    rect_center = tuple(map(int, rect_center))
                    rect_width, rect_height = rect_dims
                    
                    # Add rectangle info to the image
                    cv2.putText(temp_vis_img_bgr, f"W: {rect_width:.1f}px", 
                                (rect_center[0]-40, rect_center[1]+60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(temp_vis_img_bgr, f"H: {rect_height:.1f}px", 
                                (rect_center[0]-40, rect_center[1]+80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(temp_vis_img_bgr, f"Angle: {rect_angle:.1f}°", 
                                (rect_center[0]-40, rect_center[1]+100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Log rectangle info
                    self.get_logger().info(f"Rectangle fit: Center={rect_center}, W×H={rect_width:.1f}×{rect_height:.1f}px, Angle={rect_angle:.2f}°")
                    
                    # Draw and label the rectangle corners
                    corner_colors = [
                        (0, 255, 255),  # Yellow - top-left
                        (0, 165, 255),  # Orange - top-right  
                        (255, 0, 255),  # Magenta - bottom-left
                        (255, 255, 0)   # Cyan - bottom-right
                    ]
                    corner_labels = ["TL", "TR", "BL", "BR"]
                    corner_size = 5
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    
                    # Sort box points to meaningful order (top-left, top-right, bottom-right, bottom-left)
                    # Sort first by y (top to bottom)
                    sorted_by_y = box_points[np.argsort(box_points[:, 1])]
                    # Get top two and bottom two points
                    top_points = sorted_by_y[:2]
                    bottom_points = sorted_by_y[2:]
                    # Sort top points by x (left to right)
                    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
                    # Sort bottom points by x (left to right)
                    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
                    
                    # Collect ordered corners
                    ordered_corners = [top_left, top_right, bottom_left, bottom_right]
                    
                    # Draw and label corners
                    for i, corner in enumerate(ordered_corners):
                        corner_point = tuple(corner)
                        # Draw colored circle
                        cv2.circle(temp_vis_img_bgr, corner_point, radius=corner_size, color=corner_colors[i], thickness=-1)
                        
                        # Label position adjustment based on corner position
                        if i == 0:  # Top-left
                            label_pos = (corner_point[0]-15, corner_point[1]-5)
                        elif i == 1:  # Top-right
                            label_pos = (corner_point[0]+5, corner_point[1]-5)
                        elif i == 2:  # Bottom-left
                            label_pos = (corner_point[0]-15, corner_point[1]+15)
                        else:  # Bottom-right 
                            label_pos = (corner_point[0]+5, corner_point[1]+15)
                        
                        # Add label text
                        cv2.putText(temp_vis_img_bgr, corner_labels[i], label_pos, font, font_scale, corner_colors[i], 2)
                    
                    # Log corner positions
                    self.get_logger().info("------ Rectangle Corner Pixels (Image Coordinates) ------")
                    self.get_logger().info(f"Top-Left:     x={top_left[0]}, y={top_left[1]}")
                    self.get_logger().info(f"Top-Right:    x={top_right[0]}, y={top_right[1]}")
                    self.get_logger().info(f"Bottom-Left:  x={bottom_left[0]}, y={bottom_left[1]}")
                    self.get_logger().info(f"Bottom-Right: x={bottom_right[0]}, y={bottom_right[1]}")
                    self.get_logger().info("-----------------------------------------------")
                    
                    # Convert back to RGB
                    visualized_image = cv2.cvtColor(temp_vis_img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    self.get_logger().warn("No inlier pixel coordinates available for highlighting.")

                # Add marker for tallest object (using the 2D centroid from LangSAM)
                if tallest_object_id < len(response.centroids):
                    img_centroid = response.centroids[tallest_object_id]
                    cx = int(img_centroid.x)
                    cy = int(img_centroid.y)
                    # Ensure centroid is within bounds before drawing
                    # if 0 <= cx < img_w and 0 <= cy < img_h:
                    #     cv2.drawMarker(visualized_image, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_STAR,
                    #                 markerSize=30, thickness=3)
                    #     cv2.putText(visualized_image, "TALLEST", (cx+15, cy-15),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Visualize the final aligned gripper axes and estimated box corners
                #visualized_image = self.visualize_box_alignment(visualized_image, tallest_face_info)

            else:
                self.get_logger().warn('No objects with a valid aligned top face were found.')

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
                 # Fallback to OpenCV save
                 try:
                      cv2.imwrite(save_path, cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
                      self.get_logger().info(f'Saved result with highlighted inliers to {save_path} using OpenCV.')
                 except Exception as e_cv_save:
                      self.get_logger().error(f"Failed to save image using OpenCV as well: {e_cv_save}")


            # --- OpenCV Display Window ---
            try:
                 cv_display_image = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR)
                 # Resize for display if needed
                 max_disp_w, max_disp_h = 1280, 720
                 h, w = cv_display_image.shape[:2]
                 if w > max_disp_w or h > max_disp_h:
                      scale = min(max_disp_w/w, max_disp_h/h)
                      nw, nh = int(w*scale), int(h*scale)
                      cv_display_image = cv2.resize(cv_display_image, (nw, nh), interpolation=cv2.INTER_AREA)

                 cv2.imshow('Result PCA Aligned + Inliers', cv_display_image)
                 cv2.waitKey(1) # Essential for window updates
            except Exception as e_cv:
                 self.get_logger().warn(f"Could not display image with OpenCV: {e_cv}")


            # --- Shutdown Logic ---
            self.get_logger().info("Processing finished for this frame.")
            if self.display_time > 0:
                self.get_logger().info(f"Waiting {self.display_time} seconds before shutdown...")
                # Start timer only if not shutting down already
                if self.shutdown_timer is None or not self.shutdown_timer.is_alive():
                     self.shutdown_timer = threading.Timer(self.display_time, self.shutdown_node)
                     self.shutdown_timer.daemon = True # Allow program exit even if timer thread remains
                     self.shutdown_timer.start()
            elif self.display_time == 0:
                self.get_logger().info("Display time is 0. Node will continue running. Press Ctrl+C to exit.")
            else: # display_time < 0, shutdown immediately
                 self.get_logger().info("Display time is negative. Shutting down immediately.")
                 # Use a short delay to allow logs to flush before shutdown
                 threading.Timer(0.1, self.shutdown_node).start()


        except Exception as e:
            self.get_logger().error(f'Critical error in process_response_callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            # Use a short delay to allow logs to flush before shutdown
            threading.Timer(0.1, self.shutdown_node).start() # Shutdown on critical error
        finally:
            # Reset processing flag *after* handling is complete or on error
            # Crucial to allow next image pair processing
            self.processing_started = False
            self.get_logger().debug("Processing flag reset.")


    # --- Point Cloud and Coordinate Transforms ---

    def get_points_from_mask(self, mask, depth_image_in):
        """
        Extract 3D points AND original (x, y) pixel coords from depth image.

        Args:
            mask: Binary mask indicating object pixels.
            depth_image_in: The depth image (numpy array) to use.

        Returns:
            Tuple: (points_3d_robot, original_pixels_xy) or None if failed.
                   points_3d_robot: Nx3 numpy array in robot base frame.
                   original_pixels_xy: Nx2 numpy array of [x, y] pixel coordinates.
        """
        if depth_image_in is None:
            self.get_logger().warn("get_points_from_mask: No depth image provided.")
            return None
        if mask is None:
             self.get_logger().warn("get_points_from_mask: No mask provided.")
             return None

        depth_height, depth_width = depth_image_in.shape
        mask_height, mask_width = mask.shape

        # Ensure mask and depth image are compatible
        mask_resized = mask
        if depth_height != mask_height or depth_width != mask_width:
            self.get_logger().warn(f"Resizing mask ({mask_width}x{mask_height}) to match depth image ({depth_width}x{depth_height})")
            try:
                mask_resized = cv2.resize(mask, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
            except Exception as e_resize:
                 self.get_logger().error(f"Failed to resize mask: {e_resize}")
                 return None

        # Get pixel indices (y, x) where the mask is valid (non-zero)
        y_indices, x_indices = np.nonzero(mask_resized)
        if len(y_indices) == 0:
            self.get_logger().debug("Mask is empty after potential resize.")
            return None

        # Get depth values at these specific pixel locations
        depth_values_mm = depth_image_in[y_indices, x_indices]

        # Filter based on valid depth range (e.g., 10cm to 8m in millimeters)
        min_depth_mm = 100
        max_depth_mm = 8000
        valid_depth_indices = (depth_values_mm > min_depth_mm) & (depth_values_mm < max_depth_mm)

        # Keep only the valid pixel coordinates and depths
        valid_x = x_indices[valid_depth_indices]
        valid_y = y_indices[valid_depth_indices]
        valid_depths_mm = depth_values_mm[valid_depth_indices]

        if len(valid_depths_mm) == 0:
            self.get_logger().debug(f"No valid depth readings ({min_depth_mm}-{max_depth_mm}mm) found within the mask region.")
            return None

        # --- Store the original pixel coordinates ---
        original_pixels_xy = np.column_stack((valid_x, valid_y))

        # --- Calculate 3D points in camera frame ---
        depths_m = valid_depths_mm / 1000.0
        X_cam = (valid_x - self.cx) * depths_m / self.fx
        Y_cam = (valid_y - self.cy) * depths_m / self.fy
        Z_cam = depths_m
        points_3d_camera = np.column_stack((X_cam, Y_cam, Z_cam))

        # --- Transform to robot base frame ---
        points_3d_robot = self.transform_points_to_robot_base(points_3d_camera)

        if points_3d_robot is None or len(points_3d_robot) == 0:
             self.get_logger().warn("Failed to transform points to robot base.")
             return None

        self.get_logger().debug(f"Extracted {len(points_3d_robot)} valid 3D points with corresponding pixels.")
        return points_3d_robot, original_pixels_xy


    def transform_points_to_robot_base(self, points_3d_camera):
        """Transform points from camera frame to robot base frame."""
        # IMPORTANT: VERIFY this static transform matrix for your setup!
        # This maps camera_color_optical_frame -> ur10_base_link (example)
        transform_matrix = np.array([
            [0.995, 0.095, 0.007, -0.510],
            [0.095, -0.995, -0.005, 0.010],
            [0.006, 0.006, -1.000, 1.110],
            [0.000, 0.000, 0.000, 1.000]
        ])

        if points_3d_camera is None or len(points_3d_camera) == 0:
             return None
        points_3d_camera = np.asarray(points_3d_camera)
        if points_3d_camera.ndim == 1:
             points_3d_camera = points_3d_camera.reshape(1,-1)

        n_points = points_3d_camera.shape[0]
        # Add homogeneous coordinate (column of ones)
        points_homogeneous = np.hstack((points_3d_camera, np.ones((n_points, 1))))
        # Apply transformation: P_base_hom = P_cam_hom @ T_cam_to_base^T
        transformed_points_homogeneous = points_homogeneous @ transform_matrix.T
        # Return only the x, y, z coordinates
        return transformed_points_homogeneous[:, :3]


    def transform_points_to_camera(self, points_3d_robot):
        """Transform points from robot base frame back to camera frame."""
        # Use the same transform matrix as transform_points_to_robot_base
        transform_matrix = np.array([
            [0.995, 0.095, 0.007, -0.510],
            [0.095, -0.995, -0.005, 0.010],
            [0.006, 0.006, -1.000, 1.110],
            [0.000, 0.000, 0.000, 1.000]
        ])

        try:
            # Calculate inverse transformation: T_base_to_cam
            inverse_transform = np.linalg.inv(transform_matrix)
        except np.linalg.LinAlgError:
            self.get_logger().error("Transform matrix is singular, cannot invert.")
            return None

        if points_3d_robot is None or len(points_3d_robot) == 0:
            return None
        points_3d_robot = np.asarray(points_3d_robot)
        if points_3d_robot.ndim == 1:
             points_3d_robot = points_3d_robot.reshape(1,-1)

        n_points = points_3d_robot.shape[0]
        # Add homogeneous coordinate
        points_homogeneous = np.hstack((points_3d_robot, np.ones((n_points, 1))))
        # Apply inverse transformation: P_cam_hom = P_base_hom @ T_base_to_cam^T
        transformed_points_homogeneous = points_homogeneous @ inverse_transform.T
        # Return x, y, z in camera frame
        return transformed_points_homogeneous[:, :3]


    def transform_points_to_gripper_frame(self, points, gripper_position, gripper_quaternion):
        """Transforms points from robot base frame to gripper frame."""
        if points is None or len(points) == 0:
            return np.array([])
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Calculate rotation matrix from gripper frame to base frame
        rot_matrix_gripper_to_base = self.get_rotation_matrix_from_quaternion(gripper_quaternion)
        # Inverse rotation: from base frame to gripper frame (transpose for orthogonal matrix)
        rot_matrix_base_to_gripper = rot_matrix_gripper_to_base.T

        # Translate points relative to gripper origin (in base frame)
        translated_points = points - gripper_position

        # Rotate translated points into gripper frame
        # P_gripper = R_base_to_gripper * P_translated_base
        transformed_points = translated_points @ rot_matrix_base_to_gripper.T

        return transformed_points


    # --- PCA and Plane Fitting ---

    def perform_2d_pca(self, points_2d):
        """Performs PCA on a set of 2D points."""
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
             self.get_logger().warn(f"PCA needs > 1 point for covariance calculation, got {centered_points.shape[0]}.")
             return mean, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

        # rowvar=False because each row is an observation (point), columns are variables (x, y)
        cov_matrix = np.cov(centered_points, rowvar=False)

        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            self.get_logger().error("Covariance matrix contains NaN or Inf values.")
            return mean, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

        try:
            # Use eigh for symmetric matrices like covariance matrices
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        except np.linalg.LinAlgError:
             self.get_logger().error("PCA failed: Eigenvalue decomposition failed.")
             return mean, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices] # Columns are eigenvectors

        principal_axis_1 = eigenvectors[:, 0]
        principal_axis_2 = eigenvectors[:, 1]

        self.get_logger().debug(f"PCA: Axis1={principal_axis_1}, Axis2={principal_axis_2}, Values={eigenvalues}")
        return mean, principal_axis_1, principal_axis_2, eigenvalues


    def calculate_wrist_angle_from_pca(self, principal_axis_1):
        """
        Calculates the required wrist rotation angle to align gripper's X-axis
        with the primary principal axis from PCA.
        """
        if principal_axis_1 is None: return 0.0
        # Angle of the principal axis vector (pca_axis1[0], pca_axis1[1]) relative to the positive X-axis ([1, 0])
        angle = np.arctan2(principal_axis_1[1], principal_axis_1[0])
        self.get_logger().debug(f"PCA alignment angle raw: {np.degrees(angle):.2f} deg")
        return angle


    def get_plane_info_ransac(self, points_3d, iterations=100, threshold=0.01):
        """
        Calculates plane using RANSAC. Returns dict including inlier_indices relative to input points_3d.
        """
        points_array = np.asarray(points_3d)
        num_points = len(points_array)

        if num_points < 3:
            self.get_logger().debug(f"RANSAC needs >= 3 points, got {num_points}.")
            return None

        best_inliers_count = -1
        best_plane_params = None
        best_inlier_indices_relative = None # Indices relative to points_array

        for _ in range(iterations):
            try:
                # Sample 3 unique indices
                sample_indices = random.sample(range(num_points), 3)
                sample_points = points_array[sample_indices]
            except ValueError:
                continue # Should not happen if num_points >= 3

            p1, p2, p3 = sample_points[0], sample_points[1], sample_points[2]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)

            if normal_norm < 1e-6: continue # Collinear points, skip iteration

            normal = normal / normal_norm
            # Plane equation: ax + by + cz + d = 0 => normal . p + d = 0
            # d = -normal . p1
            d = -np.dot(normal, p1)

            # Calculate distances of all points to the plane
            # distance = | normal . p + d |
            distances = np.abs(points_array @ normal + d)

            # Find inliers within the threshold
            current_inlier_indices = np.where(distances < threshold)[0]
            current_inliers_count = len(current_inlier_indices)

            # Update best model if current one is better
            if current_inliers_count > best_inliers_count:
                best_inliers_count = current_inliers_count
                best_plane_params = (normal, d)
                best_inlier_indices_relative = current_inlier_indices

                # Optional early exit heuristic
                if best_inliers_count > num_points * 0.8: # If 80% are inliers
                    self.get_logger().debug("RANSAC early exit due to high inlier count.")
                    break

        # Check if a suitable model was found
        min_ransac_inliers = 10 # Minimum number of inliers required to accept plane
        if best_plane_params is None or best_inliers_count < min_ransac_inliers:
            self.get_logger().debug(f"RANSAC failed to find plane with >= {min_ransac_inliers} inliers (best: {best_inliers_count}).")
            return None

        # Retrieve best parameters and calculate center
        final_normal, _ = best_plane_params
        final_inlier_points = points_array[best_inlier_indices_relative]
        center = np.mean(final_inlier_points, axis=0)

        result = {
            "normal": final_normal,
            "center": center,
            "inliers": final_inlier_points,
            "inlier_indices": best_inlier_indices_relative, # Indices relative to input points_3d
            "inlier_count": best_inliers_count,
            "total_points": num_points
        }
        return result


    def find_corners_using_pca_axes(self, inlier_points, center, v1_pca_3d, v2_pca_3d, percentile=5):
        """
        Finds corners based on projecting RANSAC inliers onto PCA axes using percentiles.
        """
        if inlier_points is None or len(inlier_points) < 4 or v1_pca_3d is None or v2_pca_3d is None:
            self.get_logger().warn("Insufficient data for PCA-based corner finding.")
            return {}

        # Ensure axes are normalized (should be from PCA, but double-check)
        v1_norm = np.linalg.norm(v1_pca_3d)
        v2_norm = np.linalg.norm(v2_pca_3d)
        if v1_norm < 1e-6 or v2_norm < 1e-6:
             self.get_logger().warn("PCA axes are near zero length for corner finding.")
             return {}
        v1_pca_3d = v1_pca_3d / v1_norm
        v2_pca_3d = v2_pca_3d / v2_norm

        # Project inlier points onto the PCA basis vectors relative to the center
        relative_points = inlier_points - center
        coords_v1 = relative_points @ v1_pca_3d
        coords_v2 = relative_points @ v2_pca_3d

        # Find robust min/max using percentiles
        try:
            # Use min/max of the central (100 - 2*percentile)% of points
            min_v1_robust = np.percentile(coords_v1, percentile)
            max_v1_robust = np.percentile(coords_v1, 100 - percentile)
            min_v2_robust = np.percentile(coords_v2, percentile)
            max_v2_robust = np.percentile(coords_v2, 100 - percentile)
        except IndexError: # Can happen if percentile calculation fails (e.g., too few unique values)
             self.get_logger().warn(f"Percentile calculation failed (points: {len(inlier_points)}). Falling back to min/max.")
             min_v1_robust, max_v1_robust = np.min(coords_v1), np.max(coords_v1)
             min_v2_robust, max_v2_robust = np.min(coords_v2), np.max(coords_v2)
        except ValueError as e: # Can happen with NaN/Inf
             self.get_logger().error(f"ValueError during percentile calculation: {e}. Falling back to min/max.")
             min_v1_robust, max_v1_robust = np.min(coords_v1), np.max(coords_v1)
             min_v2_robust, max_v2_robust = np.min(coords_v2), np.max(coords_v2)


        # Define corners in 3D using the robust coordinates and PCA axes
        # Order: bl, br, tr, tl relative to the PCA axes v1, v2
        corners_3d = np.array([
            center + min_v1_robust * v1_pca_3d + min_v2_robust * v2_pca_3d,
            center + max_v1_robust * v1_pca_3d + min_v2_robust * v2_pca_3d,
            center + max_v1_robust * v1_pca_3d + max_v2_robust * v2_pca_3d,
            center + min_v1_robust * v1_pca_3d + max_v2_robust * v2_pca_3d
        ])

        # Calculate edges (vectors between consecutive corners)
        edges = [corners_3d[(i + 1) % 4] - corners_3d[i] for i in range(4)]

        self.get_logger().debug(f"PCA Corners Robust Extents: v1=[{min_v1_robust:.3f}, {max_v1_robust:.3f}], v2=[{min_v2_robust:.3f}, {max_v2_robust:.3f}]")

        return {
            "corners": corners_3d,
            "edges": edges,
            "pca_basis_v1": v1_pca_3d,
            "pca_basis_v2": v2_pca_3d
        }


    def find_top_face_with_pca_alignment(self, points_3d, iterations=100, threshold=0.01):
        """
        Finds top face using RANSAC, calculates PCA-aligned gripper orientation,
        and finds corners using PCA axes.
        """
        if points_3d is None or len(points_3d) < 3:
            self.get_logger().warn("find_top_face: Not enough points provided.")
            return None

        # Minimum number of inliers required to consider a plane valid
        min_required_inliers = max(25, int(len(points_3d) * 0.1)) # Increased min inliers

        # --- 1. Find Best Candidate Top Plane using RANSAC ---
        # Run RANSAC once but more thoroughly
        best_plane_info = self.get_plane_info_ransac(points_3d, iterations * 3, threshold)

        if not (best_plane_info and best_plane_info["inlier_count"] >= min_required_inliers):
            self.get_logger().warn("RANSAC did not find a plane with sufficient inliers.")
            return None

        # Check if the best plane found is a "top face"
        normal = best_plane_info["normal"]
        # Ensure normal points upwards (positive Z in robot base frame)
        if normal[2] < 0:
            best_plane_info["normal"] = -normal
            normal = best_plane_info["normal"]

        up_vector = np.array([0, 0, 1])
        alignment_with_up = np.dot(normal, up_vector)
        min_up_alignment = 0.7 # How much the normal must point upwards

        if alignment_with_up < min_up_alignment:
            self.get_logger().info(f"Best plane found, but normal {np.round(normal,3)} not pointing up enough (align={alignment_with_up:.3f} < {min_up_alignment}).")
            return None

        # Plane is considered a valid top face
        self.get_logger().info(f"Selected Top Plane: Z={best_plane_info['center'][2]:.3f}, Inliers={best_plane_info['inlier_count']}")

        # --- 2. Extract Info ---
        center = best_plane_info["center"]
        inlier_points = best_plane_info["inliers"]
        inlier_indices_relative_to_input = best_plane_info["inlier_indices"]

        # --- 3. Calculate Initial Gripper Orientation ---
        # Gripper Z-axis points opposite to surface normal for top grasp
        z_axis_gripper = -normal
        # Define initial X/Y axes in base frame, orthogonal to Z, trying to align X with base X
        global_x_base = np.array([1.0, 0.0, 0.0])
        # Calculate initial Y axis: y = z x x (Right Hand Rule)
        initial_y_axis = np.cross(z_axis_gripper, global_x_base)
        if np.linalg.norm(initial_y_axis) < 1e-5: # If z_axis parallel to global_x
            global_y_base = np.array([0.0, 1.0, 0.0])
            initial_y_axis = np.cross(z_axis_gripper, global_y_base) # Use global Y instead
        # Normalize initial Y axis
        y_norm = np.linalg.norm(initial_y_axis)
        if y_norm < 1e-5: # Degenerate case (normal aligned with Z axis)
             self.get_logger().warn("Normal is aligned with Z-axis. Using standard XY for initial gripper frame.")
             initial_x_axis = np.array([1., 0., 0.])
             initial_y_axis = np.array([0., 1., 0.])
        else:
             initial_y_axis /= y_norm
             # Calculate initial X axis: x = y x z
             initial_x_axis = np.cross(initial_y_axis, z_axis_gripper)
             # initial_x_axis should be normalized already

        # Create initial rotation matrix (columns are axes in base frame)
        initial_rot_matrix = np.eye(4)
        initial_rot_matrix[:3, 0] = initial_x_axis
        initial_rot_matrix[:3, 1] = initial_y_axis
        initial_rot_matrix[:3, 2] = z_axis_gripper
        # Convert matrix to quaternion [x, y, z, w]
        initial_gripper_quaternion = quaternion_from_matrix(initial_rot_matrix)
        self.get_logger().debug(f"Initial Gripper Quat: {np.round(initial_gripper_quaternion, 3)}")


        # --- 4. Transform Inliers to Initial Gripper Frame for PCA ---
        points_in_initial_gripper_frame = self.transform_points_to_gripper_frame(
            inlier_points, center, initial_gripper_quaternion
        )
        if points_in_initial_gripper_frame is None or len(points_in_initial_gripper_frame) == 0:
             self.get_logger().error("Failed to transform points to initial gripper frame for PCA.")
             # Return basic info without PCA alignment or corners
             return { "center": center, "normal": normal, "inliers": inlier_points,
                      "inlier_indices": inlier_indices_relative_to_input,
                      "final_gripper_quaternion": initial_gripper_quaternion, # Fallback
                      "wrist_angle": 0.0, "inlier_count": len(inlier_points) }

        # Project onto XY plane of the initial gripper frame
        points_2d_projected = points_in_initial_gripper_frame[:, :2]


        # --- 5. Perform PCA on Projected Points ---
        pca_mean_2d, pca_axis1_2d, pca_axis2_2d, pca_eigenvalues = self.perform_2d_pca(points_2d_projected)


        # --- 6. Calculate Wrist Angle and Final Orientation ---
        final_gripper_quaternion = initial_gripper_quaternion # Default if PCA fails
        wrist_angle = 0.0
        v1_pca_3d = None # Initialize 3D PCA axes in base frame
        v2_pca_3d = None

        if pca_axis1_2d is not None and pca_axis2_2d is not None:
            # Calculate wrist angle needed to align initial gripper X with pca_axis1_2d
            wrist_angle = self.calculate_wrist_angle_from_pca(pca_axis1_2d)
            self.get_logger().info(f"Calculated PCA alignment (wrist) angle: {np.degrees(wrist_angle):.2f} degrees")

            # Calculate final orientation by applying wrist rotation
            # Rotation is around the Z-axis *of the initial gripper frame* ([0, 0, 1])
            rotation_around_z = quaternion_about_axis(wrist_angle, [0, 0, 1])
            # Final orientation = Initial orientation * Local Rotation
            final_gripper_quaternion = quaternion_multiply(initial_gripper_quaternion, rotation_around_z)
            final_gripper_quaternion /= np.linalg.norm(final_gripper_quaternion) # Normalize
            self.get_logger().debug(f"Final Gripper Quat: {np.round(final_gripper_quaternion, 3)}")

            # --- Calculate the 3D PCA axes in the robot base frame ---
            # These represent the directions of the box edges on the plane
            # Use the initial gripper axes (in base frame) and the 2D PCA vectors
            v1_pca_3d = pca_axis1_2d[0] * initial_x_axis + pca_axis1_2d[1] * initial_y_axis
            v2_pca_3d = pca_axis2_2d[0] * initial_x_axis + pca_axis2_2d[1] * initial_y_axis
            # Normalize just in case
            v1_norm = np.linalg.norm(v1_pca_3d)
            v2_norm = np.linalg.norm(v2_pca_3d)
            if v1_norm > 1e-6: v1_pca_3d /= v1_norm
            if v2_norm > 1e-6: v2_pca_3d /= v2_norm

        else:
            self.get_logger().warn("PCA failed. Using initial orientation and cannot find PCA-based corners.")


        # --- 7. Find Corners using PCA axes (if available) ---
        corners_info = {} # Default to empty dictionary
        if v1_pca_3d is not None and v2_pca_3d is not None:
             self.get_logger().info("Finding corners using robust PCA axes.")
             # Use a percentile (e.g., 5) to define robust min/max extent
             corners_info = self.find_corners_using_pca_axes(
                 inlier_points, center, v1_pca_3d, v2_pca_3d, percentile=5
             )
        else:
             self.get_logger().warn("Skipping PCA corner finding as PCA axes are not available.")


        # --- 8. Assemble Final Information ---
        result_info = {
            "center": center,
            "normal": normal,
            "inliers": inlier_points, # RANSAC inlier 3D points
            "inlier_indices": inlier_indices_relative_to_input, # Indices relative to original points_3d input
            "inlier_count": best_plane_info["inlier_count"],
            "initial_gripper_quaternion": initial_gripper_quaternion,
            "final_gripper_quaternion": final_gripper_quaternion, # PCA aligned orientation
            "wrist_angle": wrist_angle, # Angle derived from PCA
            "pca_mean_2d": pca_mean_2d, # Centroid of 2D projected points
            "pca_axis1_2d": pca_axis1_2d, # Primary PCA direction in 2D projection
            "pca_axis2_2d": pca_axis2_2d, # Secondary PCA direction in 2D projection
            "pca_eigenvalues": pca_eigenvalues,
            "corners": corners_info.get("corners", None), # 4x3 array or None
            "edges": corners_info.get("edges", None),     # List of 4 edge vectors or None
            "pca_basis_v1": corners_info.get("pca_basis_v1", None), # 3D vector used for corners
            "pca_basis_v2": corners_info.get("pca_basis_v2", None), # 3D vector used for corners
        }

        return result_info


    # --- Visualization and Utility functions ---

    def calculate_edge_positions_in_gripper_frame(self, plane_info):
        """Calculates and logs box edge info relative to the FINAL gripper frame."""
        if plane_info.get("corners") is None:
            # Don't log if no corners were found
            return

        corners_base = plane_info["corners"]
        final_gripper_pos = plane_info["center"]
        final_gripper_quat = plane_info["final_gripper_quaternion"] # Use the PCA-aligned one

        # Transform corners from base frame to the final gripper frame
        corners_gripper_frame = self.transform_points_to_gripper_frame(
            corners_base, final_gripper_pos, final_gripper_quat
        )

        if corners_gripper_frame is None or len(corners_gripper_frame) < 4:
            self.get_logger().warn("Failed to transform corners to gripper frame or < 4 corners.")
            return

        self.get_logger().info("------ Box Edges in Final Gripper Frame (PCA Aligned) ------")
        self.get_logger().info("  Corners in final gripper frame (X,Y should define plane, Z~0):")
        for i, corner in enumerate(corners_gripper_frame):
            self.get_logger().info(f"    Corner {i}: X={corner[0]:.4f}, Y={corner[1]:.4f}, Z={corner[2]:.4f}")

        # Analyze edges in the gripper XY plane
        self.get_logger().info("  Edges projected onto final gripper XY plane:")
        total_x_aligned, total_y_aligned = 0, 0
        edge_lengths = []
        for i in range(len(corners_gripper_frame)):
            p1 = corners_gripper_frame[i][:2] # XY projection
            p2 = corners_gripper_frame[(i + 1) % len(corners_gripper_frame)][:2]
            edge_vec = p2 - p1
            length = np.linalg.norm(edge_vec)
            edge_lengths.append(length)
            # Calculate next index for logging
            next_i = (i + 1) % len(corners_gripper_frame)
            if length > 0.01: # Ignore very short edges
                edge_norm = edge_vec / length
                # Angle with gripper X-axis ([1, 0])
                angle_deg = np.degrees(np.arctan2(edge_norm[1], edge_norm[0]))
                # Alignment scores (cosine of angle between edge and axis)
                align_x = abs(np.dot(edge_norm, [1, 0]))
                align_y = abs(np.dot(edge_norm, [0, 1]))
                # --- CORRECTED LINE 1 ---
                self.get_logger().info(f"    Edge {i}->{next_i}: Len={length:.3f}m, Angle={angle_deg:.1f}d, Align(X:{align_x:.2f}, Y:{align_y:.2f})")
                # Check alignment (allow some tolerance)
                if align_x > 0.95: total_x_aligned += 1
                elif align_y > 0.95: total_y_aligned += 1
            else:
                # --- CORRECTED LINE 2 ---
                self.get_logger().info(f"    Edge {i}->{next_i}: Near zero length ({length:.4f}m).")

        self.get_logger().info(f"  Edge Lengths (m): {[f'{l:.3f}' for l in edge_lengths]}")
        self.get_logger().info(f"  Alignment Summary: {total_x_aligned} edges align X, {total_y_aligned} edges align Y (Expected: 2+2 ideally)")
        self.get_logger().info("-----------------------------------------------------------")

    # def visualize_box_alignment(self, visualized_image, plane_info):
    #     """Visualizes the calculated corners and the FINAL aligned gripper axes."""
    #     center_3d = plane_info["center"]
    #     center_2d = self.project_point_to_image(center_3d)
    #     if center_2d is None:
    #         self.get_logger().warn("Cannot visualize alignment, center point projects outside image.")
    #         return visualized_image

    #     # --- Draw Box Corners and Edges (using PCA-derived corners) ---
    #     corners_3d = plane_info.get("corners")
    #     corners_2d = []
    #     valid_corner_projections = 0
    #     if corners_3d is not None and len(corners_3d) == 4:
    #         # Convert image to BGR for OpenCV drawing functions
    #         vis_img_bgr = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR)
    #         corner_color_bgr = (0, 255, 255) # Yellow in BGR
    #         edge_color_bgr = (0, 255, 255)

    #         for i, corner in enumerate(corners_3d):
    #             corner_2d_proj = self.project_point_to_image(corner)
    #             if corner_2d_proj is not None:
    #                 corners_2d.append(corner_2d_proj)
    #                 #cv2.circle(vis_img_bgr, corner_2d_proj, 5, corner_color_bgr, -1)
    #                 valid_corner_projections += 1
    #             else:
    #                 corners_2d.append(None) # Keep placeholder

    #         # Draw edges only if all 4 corners project validly
    #         if valid_corner_projections == 4:
    #              for i in range(4):
    #                  if corners_2d[i] is not None and corners_2d[(i + 1) % 4] is not None:
    #                      pass
    #                       #cv2.line(vis_img_bgr, corners_2d[i], corners_2d[(i + 1) % 4], edge_color_bgr, 2)
    #         else:
    #              self.get_logger().debug(f"Only {valid_corner_projections}/4 corners projected onto image, not drawing full box outline.")

    #     else:
    #         #vis_img_bgr = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR) # Still convert for axis drawing
    #         self.get_logger().debug("Corners not available or not 4 corners found for visualization.")

    #     # --- Draw Final Aligned Gripper Axes ---
    #     final_quat = plane_info["final_gripper_quaternion"]
    #     rot_matrix = self.get_rotation_matrix_from_quaternion(final_quat)
    #     x_axis, y_axis, z_axis = rot_matrix[:, 0], rot_matrix[:, 1], rot_matrix[:, 2]
    #     axis_length_m = 0.07 # Length of axes lines in meters

    #     # Calculate 3D endpoints of axes
    #     x_end_3d = center_3d + x_axis * axis_length_m
    #     y_end_3d = center_3d + y_axis * axis_length_m
    #     z_end_3d = center_3d + z_axis * axis_length_m

    #     # Project 3D endpoints to 2D image coordinates
    #     x_end_2d = self.project_point_to_image(x_end_3d)
    #     y_end_2d = self.project_point_to_image(y_end_3d)
    #     z_end_2d = self.project_point_to_image(z_end_3d)

    #     # Draw axes lines (check if projection is valid)
    #     axis_color_x_bgr = (0, 0, 255) # Red
    #     axis_color_y_bgr = (0, 255, 0) # Green
    #     axis_color_z_bgr = (255, 0, 0) # Blue
    #     axis_thickness = 2

    #     # if x_end_2d: cv2.line(vis_img_bgr, center_2d, x_end_2d, axis_color_x_bgr, axis_thickness)
    #     # if y_end_2d: cv2.line(vis_img_bgr, center_2d, y_end_2d, axis_color_y_bgr, axis_thickness)
    #     # if z_end_2d: cv2.line(vis_img_bgr, center_2d, z_end_2d, axis_color_z_bgr, axis_thickness)

    #     # Add labels near end points
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 0.6
    #     font_thickness = 1
    #     label_offset = (5, 5) # Offset text slightly from axis end
    #     # if x_end_2d: cv2.putText(vis_img_bgr, "X", (x_end_2d[0]+label_offset[0], x_end_2d[1]+label_offset[1]), font, font_scale, axis_color_x_bgr, font_thickness+1, cv2.LINE_AA)
    #     # if y_end_2d: cv2.putText(vis_img_bgr, "Y", (y_end_2d[0]+label_offset[0], y_end_2d[1]+label_offset[1]), font, font_scale, axis_color_y_bgr, font_thickness+1, cv2.LINE_AA)
    #     # if z_end_2d: cv2.putText(vis_img_bgr, "Z", (z_end_2d[0]+label_offset[0], z_end_2d[1]+label_offset[1]), font, font_scale, axis_color_z_bgr, font_thickness+1, cv2.LINE_AA)

    #     # Add Wrist Angle text
    #     wrist_angle_deg = np.degrees(plane_info["wrist_angle"])
    #     angle_text = f"Align: {wrist_angle_deg:.1f}d"
    #     # Position text near the center but slightly offset
    #     text_pos = (center_2d[0] - 50, center_2d[1] + 40)
    #     # Add a small black background for readability
    #     (text_w, text_h), _ = cv2.getTextSize(angle_text, font, font_scale, font_thickness)
    #     cv2.rectangle(vis_img_bgr, (text_pos[0]-2, text_pos[1]+2), (text_pos[0]+text_w+2, text_pos[1]-text_h-2), (0,0,0), -1)
    #     cv2.putText(vis_img_bgr, angle_text, text_pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # White text

    #     # Convert back to RGB for saving/matplotlib display
    #     visualized_image_out = cv2.cvtColor(vis_img_bgr, cv2.COLOR_BGR2RGB)

    #     return visualized_image_out


    def project_point_to_image(self, point_3d_robot):
        """Projects a 3D point in robot base frame to 2D image coordinates."""
        if point_3d_robot is None: return None

        # Transform point from robot base frame to camera frame
        point_camera_frame = self.transform_points_to_camera(np.array([point_3d_robot]))

        if point_camera_frame is None or len(point_camera_frame) == 0:
            # self.get_logger().debug("Projection failed: Cannot transform point to camera frame.")
            return None

        point_camera = point_camera_frame[0] # Extract the single point's coordinates

        # Check if point is valid (in front of camera, positive Z)
        if point_camera[2] <= 0.01: # Use a small epsilon > 0
            # self.get_logger().debug(f"Projection failed: Point Z <= 0 ({point_camera[2]:.3f}) in camera frame.")
            return None

        # Apply perspective projection formula using camera intrinsics
        u = self.fx * point_camera[0] / point_camera[2] + self.cx
        v = self.fy * point_camera[1] / point_camera[2] + self.cy

        # Round to integer pixel coordinates
        u_int, v_int = int(round(u)), int(round(v))

        # Check if projected point is within image bounds
        img_h, img_w = -1, -1
        if self.latest_rgb_image is not None:
             img_h, img_w = self.latest_rgb_image.shape[:2]
        else: # Estimate bounds if image not available
             img_h, img_w = 720, 1280 # Default common size

        if 0 <= u_int < img_w and 0 <= v_int < img_h:
            return (u_int, v_int)
        else:
            # self.get_logger().debug(f"Projection failed: Point ({u_int}, {v_int}) outside image bounds (WxH: {img_w}x{img_h}).")
            return None # Point projects outside image


    def shutdown_node(self):
        """Gracefully shuts down the node."""
        self.get_logger().info('Shutdown requested. Cleaning up...')
        # Cancel any pending shutdown timer
        if self.shutdown_timer and self.shutdown_timer.is_alive():
            self.shutdown_timer.cancel()
            self.get_logger().info('Shutdown timer cancelled.')

        # Close OpenCV window if it exists
        try:
            # Check if window exists and is visible before destroying
            if cv2.getWindowProperty('Result PCA Aligned + Inliers', cv2.WND_PROP_VISIBLE) >= 1:
                 cv2.destroyWindow('Result PCA Aligned + Inliers')
                 cv2.waitKey(50) # Give time for window to close cleanly
                 self.get_logger().info('OpenCV window closed.')
        except Exception as e_cv_destroy:
            # Don't crash if window doesn't exist or closing fails
            self.get_logger().debug(f"Error closing OpenCV window (may be harmless): {e_cv_destroy}")

        # Check if ROS context is still valid before shutting down
        if rclpy.ok():
            self.get_logger().info('Shutting down ROS context...')
            # Node destruction might be handled by shutdown, but doesn't hurt to call
            # self.destroy_node()
            rclpy.shutdown()
        else:
            self.get_logger().info('ROS context already shut down.')
        self.get_logger().info('Node shutdown sequence complete.')


# --- Main Execution ---
def main(args=None):
    """Main function to initialize and run the ROS node."""
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description='Integrated Segmenter using LangSAM and PCA Alignment')
    parser.add_argument('--prompt', type=str, default='box', help='Text prompt for LangSAM segmentation')
    parser.add_argument('--save', type=str, help='Path to save the result image (e.g., output.png)')
    parser.add_argument('--wait', type=int, default=30, help='Time (seconds) to display results before shutting down. 0 waits indefinitely.')

    # Use parse_known_args to separate ROS-specific args from script args
    parsed_args, remaining_ros_args = parser.parse_known_args(sys.argv[1:])

    segmenter_node = None # Define outside try block for finally clause
    try:
        segmenter_node = IntegratedSegmenter(
            text_prompt=parsed_args.prompt,
            save_path=parsed_args.save,
            display_time=parsed_args.wait
        )
        rclpy.spin(segmenter_node)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt received, shutting down...')
        # Shutdown is handled in the finally block
    except Exception as e:
         # Log fatal exceptions that cause spin to exit
         if segmenter_node:
              segmenter_node.get_logger().fatal(f"Unhandled exception during spin: {e}")
              import traceback
              segmenter_node.get_logger().error(traceback.format_exc())
         else:
              print(f"Unhandled exception before node initialized or during spin: {e}")
              import traceback
              traceback.print_exc()
    finally:
        print('Performing final cleanup...')
        # Ensure node exists and ROS is ok before explicit shutdown actions
        if segmenter_node is not None:
             # Initiate shutdown sequence within the node
             segmenter_node.shutdown_node()
        else:
             # If node creation failed or doesn't exist, attempt basic cleanup
             print("Node object not available for controlled shutdown. Attempting basic cleanup.")
             try:
                 cv2.destroyAllWindows() # Close any potential lingering windows
             except Exception: pass
             if rclpy.ok():
                 rclpy.shutdown() # Shutdown ROS context if still ok
        print("Cleanup finished.")


if __name__ == '__main__':
    main()

# --- END OF FILE segmentation_and_pose_estimation.py ---