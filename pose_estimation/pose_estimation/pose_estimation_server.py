#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
from sensor_msgs.msg import Image
import numpy as np
import random
from lang_sam_interfaces.srv import SegmentImage
from pose_estimation_interfaces.srv import PoseEstimation
from geometry_msgs.msg import Point, Quaternion
import time
import cv2
import os

class RANSACSegmentationService(Node):
    def __init__(self):
        super().__init__('ransac_segmentation_server')
        
        # Create callback groups for concurrent execution
        self.service_group = ReentrantCallbackGroup()
        self.subscription_group = MutuallyExclusiveCallbackGroup()
        self.client_group = ReentrantCallbackGroup()
        
        # Initialize service client for LangSAM
        self.lang_sam_client = self.create_client(
            SegmentImage, 
            'segment_image',
            callback_group=self.client_group
        )
        
        # Create the service server
        self.service = self.create_service(
            PoseEstimation, 
            'estimate_pose',
            self.service_callback,
            callback_group=self.service_group
        )
        
        # Image storage with threading lock
        self.image_lock = threading.Lock()
        self.rgb_image = None
        self.depth_image = None
        self.rgb_received = False
        self.depth_received = False
        self.images_ready = False
        self.depth_buffer = []
        self.depth_buffer_size = 20  # Number of frames to average
        self.depth_buffer_lock = threading.Lock()

        # Image subscriptions
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.rgb_callback,
            rclpy.qos.qos_profile_sensor_data,
            callback_group=self.subscription_group
        )
            
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            rclpy.qos.qos_profile_sensor_data,
            callback_group=self.subscription_group
        )
        
        # SAM service status
        self.sam_connected = False
        self.sam_connection_checked = False
        self.use_dummy_sam = False
        
        # Camera intrinsics (VERIFY THESE VALUES FOR YOUR CAMERA)
        self.fx = 641.415771484375
        self.fy = 640.7596435546875
        self.cx = 650.3182983398438
        self.cy = 357.72979736328125
        
        # Visualization settings
        self.visualization_dir = "/tmp/ransac_segmentation_viz"
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        self.get_logger().info('=========================================')
        self.get_logger().info('RANSAC Segmentation Service initialized')
        self.get_logger().info('This service performs RANSAC plane fitting on segmented objects')
        self.get_logger().info(f'Visualizations will be saved to {self.visualization_dir}')
        self.get_logger().info('=========================================')
        
        # Start a thread to check SAM service availability
        threading.Thread(target=self.check_sam_service, daemon=True).start()
    
    def check_sam_service(self):
        """Check if SAM service is available in a non-blocking way."""
        self.get_logger().info("Checking SAM service availability...")
        
        # Try to wait for service with a timeout
        available = self.lang_sam_client.wait_for_service(timeout_sec=5.0)
        
        if available:
            self.get_logger().info("SAM service is AVAILABLE!")
            self.sam_connected = True
        else:
            self.get_logger().warn("SAM service is NOT available! Will use dummy responses for testing.")
            self.use_dummy_sam = True
        
        self.sam_connection_checked = True
    
    def rgb_callback(self, msg):
        """RGB image callback."""
        if self.rgb_received:
            return  # Already have an image
            
        with self.image_lock:
            try:
                if msg.encoding != 'rgb8':
                    self.get_logger().warn(f'Expected rgb8 encoding, got {msg.encoding}. Attempting conversion.')
                    try:
                        import cv_bridge
                        bridge = cv_bridge.CvBridge()
                        self.rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
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
                            self.rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                        else:
                            self.get_logger().error(f"Cannot process encoding {msg.encoding} without cv_bridge or known fallback.")
                            return
                else:
                    image_data = np.frombuffer(msg.data, dtype=np.uint8)
                    self.rgb_image = image_data.reshape((msg.height, msg.width, 3))
                
                self.rgb_received = True
                self.get_logger().info('RGB image received and cached.')
                
                # Check if both images are received
                self.check_images_ready()
            except Exception as e:
                self.get_logger().error(f'Error processing RGB image: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
    
    def depth_callback(self, msg):
        """Depth image callback that collects multiple frames for averaging."""
        if self.depth_received:
            return
            
        with self.depth_buffer_lock:
            try:
                if msg.encoding != '16UC1':
                    self.get_logger().error(f'Unsupported depth encoding: {msg.encoding}')
                    return

                depth_data = np.frombuffer(msg.data, dtype=np.uint16)
                depth_image = depth_data.reshape((msg.height, msg.width))
                
                # Add to buffer
                self.depth_buffer.append(depth_image)
                self.get_logger().info(f'Depth frame added to buffer ({len(self.depth_buffer)}/{self.depth_buffer_size})')
                
                # Process buffer when we have enough frames
                if len(self.depth_buffer) >= self.depth_buffer_size:
                    self.process_depth_buffer()
            except Exception as e:
                self.get_logger().error(f'Error processing depth image: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
    def process_depth_buffer(self):
        """Process collected depth frames with outlier rejection - faster vectorized version."""
        if len(self.depth_buffer) < 3:
            self.get_logger().warn('Not enough depth frames for averaging')
            return
            
        self.get_logger().info(f'Processing {len(self.depth_buffer)} depth frames with outlier rejection')
        
        # Stack all frames into a 3D array
        depth_stack = np.stack(self.depth_buffer, axis=0)
        
        # Create mask of valid values (> 0)
        valid_mask = depth_stack > 0
        
        # Initialize output array
        avg_depth = np.zeros_like(self.depth_buffer[0])
        
        # Count valid values per pixel
        valid_count = np.sum(valid_mask, axis=0)
        
        # Handle single valid value case (fast path)
        single_valid = valid_count == 1
        if np.any(single_valid):
            # Extract the one valid value for each pixel that has exactly one valid reading
            for i in range(len(depth_stack)):
                mask = valid_mask[i] & single_valid
                avg_depth[mask] = depth_stack[i, mask]
        
        # Handle multiple valid values case (need outlier rejection)
        multi_valid = valid_count > 1
        
        if np.any(multi_valid):
            # Calculate median for pixels with multiple valid values
            # Replace invalid values with NaN for median calculation
            temp_stack = np.where(valid_mask, depth_stack, np.nan)
            # Add this before line 211
            all_nan_pixels = np.all(np.isnan(temp_stack), axis=0)
            median_values = np.zeros_like(self.depth_buffer[0], dtype=float)
            median_values[~all_nan_pixels] = np.nanmedian(temp_stack[:, ~all_nan_pixels], axis=0)

            # Similarly before line 220
            
            
            # Calculate absolute deviation from median for each valid pixel
            abs_deviation = np.abs(depth_stack - np.expand_dims(median_values, axis=0))
            
            # Where data is invalid, set deviation to NaN
            abs_deviation = np.where(valid_mask, abs_deviation, np.nan)
            
            # Calculate MAD (median absolute deviation) for each pixel
            all_nan_pixels_dev = np.all(np.isnan(abs_deviation), axis=0)
            mad_values = np.zeros_like(self.depth_buffer[0], dtype=float)
            mad_values[~all_nan_pixels_dev] = np.nanmedian(abs_deviation[:, ~all_nan_pixels_dev], axis=0)
            
            
            # Where MAD is 0 or NaN, just use the median
            zero_mad = (mad_values == 0) | np.isnan(mad_values)
            avg_depth[multi_valid & zero_mad] = median_values[multi_valid & zero_mad].astype(np.uint16)
            
            # For remaining pixels, need to filter outliers
            remain_pixels = multi_valid & ~zero_mad
            
            if np.any(remain_pixels):
                # We need to process these pixel by pixel, but there should be fewer now
                y_indices, x_indices = np.where(remain_pixels)
                
                for i, j in zip(y_indices, x_indices):
                    # Get valid values for this pixel
                    valid_vals = depth_stack[:, i, j][valid_mask[:, i, j]]
                    
                    # Calculate inlier threshold
                    threshold = 2.0
                    inlier_mask = abs_deviation[:, i, j][valid_mask[:, i, j]] / mad_values[i, j] < threshold
                    
                    # Use inliers or median as fallback
                    inliers = valid_vals[inlier_mask]
                    if len(inliers) > 0:
                        avg_depth[i, j] = int(np.mean(inliers))
                    else:
                        avg_depth[i, j] = int(median_values[i, j])
        
        # Set the processed depth image and mark as received
        with self.image_lock:
            self.depth_image = avg_depth.astype(np.uint16)
            self.depth_received = True
            self.get_logger().info('Depth image averaged with outlier rejection')
            self.check_images_ready()
        
        # Clear the buffer
        self.depth_buffer.clear()

    def check_images_ready(self):
        """Check if both images are received."""
        if self.rgb_received and self.depth_received and not self.images_ready:
            self.images_ready = True
            self.get_logger().info('Both RGB and depth images received. Ready for processing.')
    
    def service_callback(self, request, response):
        """Non-blocking service callback that spawns a worker thread."""
        self.get_logger().info(f'Received service request with prompt: "{request.text_prompt}"')
        
        if not self.images_ready:
            self.get_logger().error('Images not yet available. Please try again later.')
            response.success = False
            response.error_message = 'Images not yet available. Please try again later.'
            return response
        
        # Create a thread-safe response object
        response_ready = threading.Event()
        response_data = {"success": False, "error_message": "Processing not completed"}
        
        # Create a new thread for handling this service request
        processing_thread = threading.Thread(
            target=self.process_request_threaded,
            args=(request, response_data, response_ready)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        # Wait for the thread to complete (with timeout)
        if not response_ready.wait(timeout=15.0):
            self.get_logger().error('Request processing timed out')
            response.success = False
            response.error_message = 'Processing timed out'
        else:
            # Copy thread results to response
            response.success = response_data["success"]
            if not response.success:
                response.error_message = response_data["error_message"]
            else:
                response.position.x = response_data["position"][0]
                response.position.y = response_data["position"][1]
                response.position.z = response_data["position"][2]

                response.orientations = []
                for quat in response_data["orientations"]:
                    orientation = Quaternion()
                    orientation.x = quat[0]
                    orientation.y = quat[1]
                    orientation.z = quat[2]
                    orientation.w = quat[3]
                    response.orientations.append(orientation)

                # Add box dimensions to the response
                response.x_width = response_data["x_width"]
                response.y_length = response_data["y_length"] 
                response.z_height = response_data["z_height"]
                
                self.get_logger().info(f'Successfully generated pose response')
        
        return response
    
    def process_request_threaded(self, request, response_data, response_ready):
        """Thread function to process the service request."""
        try:
            self.get_logger().info(f'Processing request for "{request.text_prompt}" in separate thread')
            
            # Make a copy of the images to avoid race conditions
            with self.image_lock:
                rgb_copy = self.rgb_image.copy() if self.rgb_image is not None else None
                cv2.imwrite('rgb.png', cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR))  # Save RGB image for debugging
                #Set last 70% rows to black
                
                rgb_copy = self.rgb_image.copy() if self.rgb_image is not None else None
                rgb_copy[int(rgb_copy.shape[0] * 0.3):, :, :] = 0

                                # Keep only middle 50% of columns, set the rest to black
                left_boundary = int(rgb_copy.shape[1] * 0.25)  # 25% from left
                right_boundary = int(rgb_copy.shape[1] * 0.75)  # 75% from left

                # Set left 25% to black
                rgb_copy[:, :left_boundary, :] = 0
                # Set right 25% to black
                rgb_copy[:, right_boundary:, :] = 0
                #keep only 70% of x direction

                # Set top 30% to black (keep bottom 70%)
                #rgb_copy[:int(rgb_copy.shape[0] * 0.3), :, :] = 0

                # Set left 50% to black (keep right 50%)
                #rgb_copy[:, :int(rgb_copy.shape[1] * 0.5), :] = 0

               
                #rgb_copy = rgb_copy[:int(rgb_copy.shape[0] * 0.3), :, :]
                depth_copy = self.depth_image.copy() if self.depth_image is not None else None
            
            if rgb_copy is None or depth_copy is None:
                self.get_logger().error("RGB or depth image unavailable for processing")
                response_data["success"] = False
                response_data["error_message"] = "Required images unavailable"
                response_ready.set()
                return
            
            # Convert RGB image to ROS message
            rgb_msg = Image()
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = "camera_color_optical_frame"
            rgb_msg.height = rgb_copy.shape[0]
            rgb_msg.width = rgb_copy.shape[1]
            rgb_msg.encoding = "rgb8"
            rgb_msg.is_bigendian = False
            rgb_msg.step = rgb_copy.shape[1] * 3
            rgb_msg.data = rgb_copy.tobytes()
            
            # Call SAM service or use dummy response
            if self.use_dummy_sam:
                self.get_logger().warn("Using dummy SAM response (service unavailable)")
                segment_response = self.simulate_sam_response()
                time.sleep(0.5)
            else:
                self.get_logger().info("Calling SAM service...")
                segment_future = self.call_langsam_service_async(rgb_msg, request.text_prompt)
                segment_response = self.wait_for_future(segment_future, timeout=10.0)
            
            # Check if we got a valid SAM response
            if segment_response is None:
                self.get_logger().error("Failed to get response from SAM service")
                response_data["success"] = False
                response_data["error_message"] = "SAM service call failed"
                response_ready.set()
                return
            
            # Log SAM response details
            self.get_logger().info(f"SAM found {len(segment_response.labels)} objects matching '{request.text_prompt}'")
            
            # Process each detected mask with RANSAC
            visualized_image = rgb_copy.copy()
            best_plane_info = None
            best_object_id = None
            
            for i in range(len(segment_response.labels)):
                label = segment_response.labels[i]
                score = segment_response.scores[i]
                
                self.get_logger().info(f'Processing Object {i}: {label} (score: {score:.2f})')
                
                # Extract mask data
                mask_data = np.frombuffer(segment_response.mask_images[i].data, dtype=np.uint8)
                mask = mask_data.reshape((segment_response.mask_images[i].height, segment_response.mask_images[i].width))
                
                # Apply mask overlay
                color = np.array([random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)])
                colored_mask = np.zeros_like(visualized_image)
                mask_bool = mask > 0
                for c in range(3):
                    colored_mask[:, :, c] = np.where(mask_bool, color[c], 0)
                
                alpha = 0.3  # Transparency
                visualized_image[mask_bool] = (visualized_image[mask_bool] * (1 - alpha) + colored_mask[mask_bool] * alpha).astype(np.uint8)
                
                # Extract 3D points from mask
                points_result = self.get_points_from_mask(mask, depth_copy)
                
                if points_result is None:
                    self.get_logger().warn(f"Could not get 3D points for object {i}")
                    continue
                    
                points_3d, original_pixels_xy = points_result
                
                # Apply RANSAC to find plane
                if points_3d is not None and len(points_3d) > 50:
                    plane_info = self.get_plane_info_ransac(points_3d)
                    
                    if plane_info is not None:
                        # Store plane info and inlier pixels
                        plane_info["inlier_pixels_xy"] = original_pixels_xy[plane_info["inlier_indices"]] if original_pixels_xy is not None else None
                        
                        # Check if this is the best plane so far (you can use different criteria)
                        if best_plane_info is None or plane_info["inlier_count"] > best_plane_info["inlier_count"]:
                            best_plane_info = plane_info
                            best_object_id = i
                            
                        self.get_logger().info(f'-> Found plane for object {i} with {plane_info["inlier_count"]} inliers')
                    else:
                        self.get_logger().info(f"-> Could not find a suitable plane for object {i}")
                else:
                    self.get_logger().info(f"-> Not enough valid 3D points for object {i}")
            
            # Highlight inliers for the best plane
            if best_plane_info is not None and best_object_id is not None:
                self.get_logger().info(f"Selected object {best_object_id} with {best_plane_info['inlier_count']} inliers as best result")
                
                # Highlight inlier pixels
                inlier_pixels = best_plane_info.get("inlier_pixels_xy")
                if inlier_pixels is not None and len(inlier_pixels) > 0:
                    self.get_logger().info(f"Highlighting {len(inlier_pixels)} inlier pixels")
                    
                    # Convert to BGR for OpenCV operations
                    temp_vis_img_bgr = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR)
                    
                    # Convert to integer coordinates
                    inlier_pixels = np.round(inlier_pixels).astype(int)
                    
                    # Filter out of bounds pixels
                    img_h, img_w = temp_vis_img_bgr.shape[:2]
                    valid_idx = (inlier_pixels[:, 0] >= 0) & (inlier_pixels[:, 0] < img_w) & \
                                (inlier_pixels[:, 1] >= 0) & (inlier_pixels[:, 1] < img_h)
                    inlier_pixels = inlier_pixels[valid_idx]
                    
                    # Draw small circles for inlier points
                    highlight_color_bgr = (255, 0, 255)  # Magenta in BGR
                    for px, py in inlier_pixels:
                        cv2.circle(temp_vis_img_bgr, (px, py), radius=1, color=highlight_color_bgr, thickness=-1)
                    
                    # Fit minimum area rectangle to inlier pixels
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
                    
                    # Draw and label corners with different colors
                    corner_colors = [
                        (0, 255, 255),  # Yellow - top-left
                        (0, 165, 255),  # Orange - top-right  
                        (255, 0, 255),  # Magenta - bottom-left
                        (255, 255, 0)   # Cyan - bottom-right
                    ]
                    corner_labels = ["TL", "TR", "BL", "BR"]
                    corner_size = 5
                    
                    for i, corner in enumerate(ordered_corners):
                        corner_point = tuple(corner)
                        # Draw colored circle
                        cv2.circle(temp_vis_img_bgr, corner_point, radius=corner_size, color=corner_colors[i], thickness=-1)
                        
                        # Label position adjustment
                        if i == 0:  # Top-left
                            label_pos = (corner_point[0]-15, corner_point[1]-5)
                        elif i == 1:  # Top-right
                            label_pos = (corner_point[0]+5, corner_point[1]-5)
                        elif i == 2:  # Bottom-left
                            label_pos = (corner_point[0]-15, corner_point[1]+15)
                        else:  # Bottom-right 
                            label_pos = (corner_point[0]+5, corner_point[1]+15)
                        
                        # Add label text
                        cv2.putText(temp_vis_img_bgr, corner_labels[i], label_pos, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_colors[i], 2)
                    
                    # Log corner positions
                    self.get_logger().info("------ Rectangle Corner Pixels ------")
                    self.get_logger().info(f"Top-Left:     x={top_left[0]}, y={top_left[1]}")
                    self.get_logger().info(f"Top-Right:    x={top_right[0]}, y={top_right[1]}")
                    self.get_logger().info(f"Bottom-Left:  x={bottom_left[0]}, y={bottom_left[1]}")
                    self.get_logger().info(f"Bottom-Right: x={bottom_right[0]}, y={bottom_right[1]}")
                    
                    # Project 2D corners to 3D plane
                    corners_2d = [top_left, top_right, bottom_left, bottom_right]
                    corners_3d = self.project_pixels_to_plane(corners_2d, best_plane_info["normal"], best_plane_info["center"])
                    
                    if corners_3d is not None:
                        self.get_logger().info("------ 3D Rectangle Corners (Robot Base Frame) ------")
                        corner_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
                        for i, corner in enumerate(corners_3d):
                            self.get_logger().info(f"{corner_names[i]}: x={corner[0]:.4f}, y={corner[1]:.4f}, z={corner[2]:.4f}")
                        
                        # Compute 3D rectangle dimensions
                        width = np.linalg.norm(corners_3d[1] - corners_3d[0])  # Top-Right to Top-Left
                        height = np.linalg.norm(corners_3d[2] - corners_3d[0]) # Bottom-Left to Top-Left
                        self.get_logger().info(f"3D Rectangle dimensions: Width={width:.4f}m, Height={height:.4f}m")
                        
                        # Add 3D dimensions to image
                        dim_text = f"3D: W={width*1000:.1f}mm H={height*1000:.1f}mm"
                        cv2.putText(temp_vis_img_bgr, dim_text, 
                                    (rect_center[0]-40, rect_center[1]+120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Convert back to RGB for final output
                    visualized_image = cv2.cvtColor(temp_vis_img_bgr, cv2.COLOR_BGR2RGB)
                
                # Generate pose from the best plane
                center = best_plane_info["center"]
                normal = best_plane_info["normal"]
                self.get_logger().info(f'Normal Vector: {normal}')
                
                # NEW CODE: GRIPPER ALIGNMENT
                # Check if we have valid 3D corners
                if 'corners_3d' in locals() and corners_3d is not None and len(corners_3d) == 4:
                    # Use our new alignment function
                    aligned_position, orientation_quaternion, wrist_angle = self.align_gripper_with_box(corners_3d, normal)
                    yaws = [0, np.pi/2, np.pi, 3*np.pi/2]
                    quaternions = []
                    for θ in yaws:
                        q = self.quat_mul(orientation_quaternion, self.q_yaw(θ))
                        q_flip = self.closest_flip_z(q)
                        quaternions.append(q_flip)

                    # quaternions now holds four [x,y,z,w] arrays
                    for i, q in enumerate(quaternions):
                        self.get_logger().info(f"Gripper yaw={np.degrees(yaws[i]):3.0f}° → quaternion {q}")

                    self.get_logger().info(f'Aligned gripper with box vertices')
                    self.get_logger().info(f'Aligned position: {aligned_position}')
                    self.get_logger().info(f'Orientation quaternion: {orientation_quaternion}')
                    self.get_logger().info(f'Wrist angle: {np.degrees(wrist_angle):.1f}°')
                    
                    # Use the aligned values
                    center = aligned_position
                else:
                    # Fallback to normal-based orientation if corners are not available
                    self.get_logger().warn('Could not find box corners for alignment, using normal-based orientation')
                    orientation_quaternion = self.normal_to_quaternion(normal)
                    wrist_angle = 0.0

                # ADD THIS NEW CODE HERE:
                # Apply an offset of 2cm along the negative normal vector direction
                # Note: We use negative normal because normal points into the surface
                offset_distance = 0.04  # 2 cm offset
                # Ensure normal is normalized
                normal_direction = normal / np.linalg.norm(normal)
                # Ensure normal points upward (negative Z in base frame is upward)
                if normal_direction[2] > 0:
                    normal_direction = -normal_direction
                else:
                    normal_direction = normal_direction
                # Apply offset - move away from surface along normal
                offset_vector = -normal_direction * offset_distance  # Negative to move away from surface
                original_center = center.copy()
                center = center + offset_vector
                self.get_logger().info(f'Applied {offset_distance*100:.1f}cm offset along normal direction')
                self.get_logger().info(f'Original position: {original_center}')
                self.get_logger().info(f'New position with offset: {center}')

                # Then continue with the original code:
                # Set response data
                response_data["success"] = True
                response_data["position"] = center.tolist()
                response_data["orientations"] = quaternions

                try:
                    log_file_path = 'pose_positions.txt'
                    with open(log_file_path, 'a') as f:
                        timestamp = self.get_clock().now().to_msg().sec
                        position_str = f"{response_data['position'][0]:.6f} {response_data['position'][1]:.6f} {response_data['position'][2]:.6f}"
                        f.write(f"{timestamp} | '{request.text_prompt}' | {position_str}\n")
                    self.get_logger().info(f"Position logged to {log_file_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to log position to file: {e}")
                
                # Calculate and add box dimensions
                # Convert 2D rectangle dimensions to millimeters for the response
                if 'rect_width' in locals() and 'rect_height' in locals():
                    # Convert from pixels to millimeters (approximate conversion based on depth)
                    # Use the 3D calculated dimensions if available
                    if 'width' in locals() and 'height' in locals():
                        # Use the 3D dimensions we calculated earlier (already in meters)
                        response_data["x_width"] = int(width * 100)  # Convert meters to mm
                        response_data["y_length"] = int(height * 100)  # Convert meters to mm
                    else:
                        # Fallback to 2D pixel measurements with a simple scaling factor
                        # This is very approximate and should be replaced with proper 3D measurement
                        avg_depth_mm = np.mean(depth_copy[mask_bool]) if 'mask_bool' in locals() else 500
                        scale_factor = avg_depth_mm / 100.0  # Simple scaling based on depth
                        response_data["x_width"] = int(max(rect_width, rect_height) * scale_factor)
                        response_data["y_length"] = int(min(rect_width, rect_height) * scale_factor)
                else:
                    # Default values if dimensions couldn't be determined
                    response_data["x_width"] = 10  # Default 10cm width
                    response_data["y_length"] = 100 # Default 10cm length
                
                # Standard height of 25cm (250mm)
                response_data["z_height"] = 20  # 25cm in millimeters
                
                self.get_logger().info(f"Box dimensions: {response_data['x_width']}mm × {response_data['y_length']}mm × {response_data['z_height']}mm")
                
                # Add visualization of gripper alignment
                try:
                    # Project 3D aligned position back to 2D image
                    center_3d = np.array([center])  # Make it 2D array for the function
                    center_2d_points = self.project_3d_to_2d(center_3d)
                    
                    if center_2d_points is not None and len(center_2d_points) > 0:
                        center_2d = center_2d_points[0]
                        
                        # Project box corners to 2D if needed
                        if 'corners_3d' in locals() and corners_3d is not None:
                            corners_2d_points = self.project_3d_to_2d(corners_3d)
                            if corners_2d_points is not None and len(corners_2d_points) == 4:
                                # Visualize gripper alignment
                                self.get_logger().info("3D Corners available")
                            else:
                                self.get_logger().warn("Could not project corners to 2D for visualization")
                        else:
                            self.get_logger().warn("No 3D corners available for visualization")
                    else:
                        self.get_logger().warn("Could not project center point to 2D")
                except Exception as e_vis:
                    self.get_logger().error(f"Error visualizing gripper: {e_vis}")
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                
                self.get_logger().info(f"Generated pose at position: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
            else:
                self.get_logger().warn("No valid planes found in any object")
                response_data["success"] = False
                response_data["error_message"] = "No valid planes found"
            
            # Save the visualization
            try:
                # Add title with prompt
                title = f"RANSAC: '{request.text_prompt}'"
                cv2.putText(
                    cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR),
                    title,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Save the image
                cv2.imwrite('box.png', cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
                self.get_logger().info("Saved visualization to box.png")
            except Exception as e_save:
                self.get_logger().error(f"Error saving visualization: {e_save}")
            
        except Exception as e:
            self.get_logger().error(f'Error in service thread: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            response_data["success"] = False
            response_data["error_message"] = f"Exception: {str(e)}"
        
        # Signal that processing is complete
        response_ready.set()
    
    # --- GRIPPER ALIGNMENT FUNCTIONS ---
    
    def align_gripper_with_box(self, corners_3d, normal):
        """
        Compute an aligned gripper pose for a detected box.

        Args:
            corners_3d: List of 4 corner points (3D) of the box 
                        [top_left, top_right, bottom_left, bottom_right].
            normal:     Normal vector of the box’s plane (3D).

        Returns:
            tuple: (position, orientation, wrist_angle)
                position:    [x, y, z] center of the box (gripper target position).
                orientation: [x, y, z, w] quaternion for gripper orientation aligned with the box.
                wrist_angle: rotation around gripper’s z-axis (radians) for any needed in-plane adjustment.
        """
        # Ensure we have exactly 4 corners
        if corners_3d is None or len(corners_3d) != 4:
            self.get_logger().error("Cannot align gripper: need 4 corners")
            return None, None, 0.0

        # Extract named corners for clarity
        top_left, top_right, bottom_left, bottom_right = corners_3d

        # Compute edge vectors along the box’s plane
        width_vector = top_right - top_left        # Vector along one box edge
        height_vector = bottom_left - top_left     # Vector along the orthogonal box edge

        # Compute the dimensions (lengths) of the box edges (for logging or reference)
        width = np.linalg.norm(width_vector)
        height = np.linalg.norm(height_vector)
        self.get_logger().info(f"Box dimensions: width={width:.4f} m, height={height:.4f} m")

        # Compute the geometric center of the box (mean of corner points)
        center = np.mean(corners_3d, axis=0)

        # Define orthonormal axes for the gripper’s frame aligned with the box
        # X-axis: along the box's width edge (normalized)
        x_axis = width_vector / np.linalg.norm(width_vector)
        # Z-axis: align with plane normal (pointing downward toward the box)
        z_axis = normal / np.linalg.norm(normal)
        if z_axis[2] > 0:
            z_axis = -z_axis  # Flip if normal is pointing upward, so gripper faces down onto the box
        # Y-axis: perpendicular to both Z and X (cross product to get orthonormal axis in plane)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        # Recompute X-axis to ensure exact orthogonality (in case original X was not perfectly perpendicular to Z)
        x_axis = np.cross(y_axis, z_axis)

        # Construct rotation matrix from the orthonormal axes
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        # Convert rotation matrix to quaternion (x, y, z, w)
        orientation_quat = self.rotation_matrix_to_quaternion(rotation_matrix)

        # For a square gripper, no additional in-plane rotation is needed to align with the box’s edges
        wrist_angle = 0.0

        # Log the results for debugging
        self.get_logger().info(f"Gripper alignment computed: center={center}, orientation_quat={orientation_quat}, wrist_angle={np.degrees(wrist_angle):.1f}°")

        return center, orientation_quat, wrist_angle


    def project_3d_to_2d(self, points_3d):
        """Project 3D points in robot base frame back to camera 2D pixels"""
        points_3d = np.asarray(points_3d)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
        
        # Add homogeneous coordinate
        points_homogeneous = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        
        # Transform from robot base to camera frame
        # Use inverse of camera_to_base_transform
        camera_transform = np.linalg.inv(self.camera_to_base_transform)
        camera_points = points_homogeneous @ camera_transform.T
        
        # Project 3D points to 2D using camera intrinsics
        pixels = []
        for point in camera_points:
            # Convert to camera coordinates
            x_cam, y_cam, z_cam = point[:3]
            
            # Skip points behind the camera
            if z_cam <= 0:
                continue
                
            # Project to pixel coordinates
            u = int(self.fx * x_cam / z_cam + self.cx)
            v = int(self.fy * y_cam / z_cam + self.cy)
            pixels.append([u, v])
        
        return np.array(pixels)

    def visualize_gripper_alignment(self, img, corners_2d, center_2d, orientation_quat, wrist_angle=0.0):
        """
        Visualize gripper alignment on the 2D image
        
        Args:
            img: RGB image to visualize on
            corners_2d: 2D box corners [top_left, top_right, bottom_left, bottom_right]
            center_2d: 2D center point
            orientation_quat: Orientation quaternion [x, y, z, w]
            wrist_angle: Gripper wrist angle in radians
        
        Returns:
            RGB image with gripper overlay
        """
        # Convert to BGR for OpenCV
        vis_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        
        # Draw the box corners and center
        for corner in corners_2d:
            cv2.circle(vis_img, tuple(map(int, corner)), 5, (0, 255, 255), -1)
        
        cx, cy = map(int, center_2d)
        cv2.circle(vis_img, (cx, cy), 7, (0, 0, 255), -1)
        
        # Convert quaternion to rotation matrix to get axes
        qx, qy, qz, qw = orientation_quat
        rot_matrix = np.zeros((3, 3))
        
        # Quaternion to rotation matrix conversion
        rot_matrix[0, 0] = 1 - 2 * (qy**2 + qz**2)
        rot_matrix[0, 1] = 2 * (qx*qy - qz*qw)
        rot_matrix[0, 2] = 2 * (qx*qz + qy*qw)
        rot_matrix[1, 0] = 2 * (qx*qy + qz*qw)
        rot_matrix[1, 1] = 1 - 2 * (qx**2 + qz**2)
        rot_matrix[1, 2] = 2 * (qy*qz - qx*qw)
        rot_matrix[2, 0] = 2 * (qx*qz - qy*qw)
        rot_matrix[2, 1] = 2 * (qy*qz + qx*qw)
        rot_matrix[2, 2] = 1 - 2 * (qx**2 + qy**2)
        
        # Apply additional wrist rotation if needed
        if abs(wrist_angle) > 0.001:
            # Create rotation matrix for wrist angle (rotation around Z-axis)
            cos_a = np.cos(wrist_angle)
            sin_a = np.sin(wrist_angle)
            wrist_rot = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            # Apply to the rotation matrix
            rot_matrix = rot_matrix @ wrist_rot
        
        # Extract axes
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        
        # Scale for visualization (assume gripper is 15 units wide)
        gripper_half_size = 7.5
        scale = 30  # Pixels per unit
        
        # Calculate gripper corners in 2D image space
        gripper_corners = []
        gripper_corners.append((cx + int(scale * (x_axis[0] * gripper_half_size + y_axis[0] * gripper_half_size)), 
                               cy + int(scale * (x_axis[1] * gripper_half_size + y_axis[1] * gripper_half_size))))
        gripper_corners.append((cx + int(scale * (x_axis[0] * gripper_half_size - y_axis[0] * gripper_half_size)), 
                               cy + int(scale * (x_axis[1] * gripper_half_size - y_axis[1] * gripper_half_size))))
        gripper_corners.append((cx + int(scale * (-x_axis[0] * gripper_half_size - y_axis[0] * gripper_half_size)), 
                               cy + int(scale * (-x_axis[1] * gripper_half_size - y_axis[1] * gripper_half_size))))
        gripper_corners.append((cx + int(scale * (-x_axis[0] * gripper_half_size + y_axis[0] * gripper_half_size)), 
                               cy + int(scale * (-x_axis[1] * gripper_half_size - y_axis[1] * gripper_half_size))))
        
        # Draw gripper outline
        for i in range(4):
            cv2.line(vis_img, gripper_corners[i], gripper_corners[(i+1)%4], (0, 255, 0), 2)
        
        # Draw axes for clarity
        axis_length = 50
        cv2.line(vis_img, (cx, cy), 
                (cx + int(x_axis[0] * axis_length), cy + int(x_axis[1] * axis_length)), 
                (255, 0, 0), 2)  # X-axis in blue
        cv2.line(vis_img, (cx, cy), 
                (cx + int(y_axis[0] * axis_length), cy + int(y_axis[1] * axis_length)), 
                (0, 255, 0), 2)  # Y-axis in green
        
        # Add legend
        cv2.putText(vis_img, "Gripper", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Wrist Angle: {np.degrees(wrist_angle):.1f}°", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    # --- RANSAC and Point Cloud Functions ---
    
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
        
        transform_matrix  = np.array([
            [0.995, 0.095, 0.007, 0.499],
            [0.095, -0.995, -0.005, 0.065],
            [0.006, 0.006, -1.000, 1.114],
            [0.000, 0.000, 0.000, 1.000]
        ])
        # Store the transform matrix as instance variable for later use
        if not hasattr(self, 'camera_to_base_transform'):
            self.camera_to_base_transform = transform_matrix

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
        transformed_points = transformed_points_homogeneous[:, :3]

        # Return only the x, y, z coordinates
        return transformed_points

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
    
    def project_pixels_to_plane(self, pixel_points, plane_normal, plane_point):
        """
        Project 2D pixel points onto the 3D plane using ray-plane intersection.
        
        Args:
            pixel_points: List of [x, y] pixel coordinates
            plane_normal: Normal vector of the plane
            plane_point: A point on the plane
        
        Returns:
            List of 3D points on the plane in robot base frame
        """
        if not pixel_points or len(pixel_points) == 0:
            return None
            
        # Plane equation: normal·(x-point) = 0
        # Calculate d in plane equation: normal·x + d = 0
        d = -np.dot(plane_normal, plane_point)
        
        projected_points_3d = []
        
        for pixel in pixel_points:
            # Unpack pixel coordinates
            u, v = pixel[0], pixel[1]
            
            # 1. Create a ray from camera center through the pixel
            # Convert pixel to camera coordinates (normalized direction)
            x_dir = (u - self.cx) / self.fx
            y_dir = (v - self.cy) / self.fy
            z_dir = 1.0  # Unit depth along camera Z axis
            
            ray_direction = np.array([x_dir, y_dir, z_dir])
            ray_origin = np.array([0.0, 0.0, 0.0])  # Camera center
            
            # 2. Transform ray to robot base frame
            # First, normalize the direction vector
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            
            # Transform ray origin (camera center) to robot base frame
            ray_origin_homogeneous = np.append(ray_origin, 1.0)
            ray_origin_base = (ray_origin_homogeneous @ self.camera_to_base_transform.T)[:3]
            
            # Transform ray direction to robot base frame (exclude translation)
            rotation_part = self.camera_to_base_transform[:3, :3]
            ray_direction_base = ray_direction @ rotation_part.T
            ray_direction_base = ray_direction_base / np.linalg.norm(ray_direction_base)
            
            # 3. Calculate intersection with the plane
            # Ray equation: p(t) = origin + t * direction
            # Plane equation: normal·p + d = 0
            # Solving for t: normal·(origin + t*direction) + d = 0
            # Therefore: t = -(normal·origin + d) / (normal·direction)
            
            denominator = np.dot(plane_normal, ray_direction_base)
            
            # Check if ray is parallel to plane (or nearly so)
            if abs(denominator) < 1e-6:
                self.get_logger().warn(f"Ray through pixel ({u}, {v}) is parallel to plane")
                continue
                
            t = -(np.dot(plane_normal, ray_origin_base) + d) / denominator
            
            # Check if intersection is behind the camera
            if t <= 0:
                self.get_logger().warn(f"Intersection behind camera for pixel ({u}, {v})")
                continue
                
            # Calculate the 3D intersection point in robot base frame
            intersection_point = ray_origin_base + t * ray_direction_base
            projected_points_3d.append(intersection_point)
        
        # Return None if no valid projections
        if not projected_points_3d:
            self.get_logger().warn("No valid pixel projections found")
            return None
            
        return np.array(projected_points_3d)
    
    # --- Service and SAM-related Functions ---
    
    def call_langsam_service_async(self, ros_img, text_prompt):
        """Asynchronous call to the LangSAM segmentation service."""
        request = SegmentImage.Request()
        request.image = ros_img
        request.text_prompt = text_prompt
        # Default confidence threshold
        request.confidence_threshold = 0.35
        
        # Check service availability
        if not self.lang_sam_client.service_is_ready():
            self.get_logger().warn('SAM service not ready, waiting briefly...')
            ready = self.lang_sam_client.wait_for_service(timeout_sec=2.0)
            if not ready:
                self.get_logger().error('SAM service not available after waiting')
                return None
        
        self.get_logger().info(f'Sending request to SAM service for prompt: "{text_prompt}"')
        
        try:
            # Use async call that doesn't block the thread
            future = self.lang_sam_client.call_async(request)
            return future
        except Exception as e:
            self.get_logger().error(f'Error starting SAM service call: {e}')
            return None
    
    def wait_for_future(self, future, timeout=10.0):
        """Wait for a future to complete with timeout, without blocking ROS spinning."""
        if future is None:
            self.get_logger().error("Cannot wait for None future")
            return None
            
        # Wait for future with timeout
        start_time = time.time()
        self.get_logger().info(f"Waiting for SAM service response (timeout: {timeout}s)...")
        
        # Print periodic progress updates while waiting
        while not future.done() and time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            if int(elapsed) % 2 == 0:  # Print every 2 seconds
                self.get_logger().info(f"Still waiting for SAM response... ({elapsed:.1f}s)")
            # Sleep briefly to avoid CPU spinning
            time.sleep(0.1)
        
        if not future.done():
            self.get_logger().error(f'SAM service did not respond within {timeout} seconds')
            return None
            
        try:
            self.get_logger().info(f"SAM future completed after {time.time() - start_time:.2f}s")
            result = future.result()
            return result
        except Exception as e:
            self.get_logger().error(f'SAM service future raised an exception: {e}')
            return None

    def simulate_sam_response(self):
        """Create a dummy SAM response for testing when real SAM service unavailable."""
        self.get_logger().info("Creating simulated SAM response...")
        
        # Import std_msgs for the Header
        import std_msgs.msg
        
        response = SegmentImage.Response()
        
        # Generate random number of objects (1-3)
        num_objects = random.randint(1, 3)
        
        response.labels = [f"object{i+1}" for i in range(num_objects)]
        response.scores = [random.uniform(0.7, 0.99) for _ in range(num_objects)]
        
        # Create fake masks for each object
        mask_images = []
        centroids = []
        
        # Use simple 480x640 masks for testing
        height, width = 480, 640
        
        for i in range(num_objects):
            # Create a mask with random pixels set
            mask = Image()
            mask.height = height
            mask.width = width
            mask.encoding = "mono8"
            
            # Create a simple circular mask pattern in different positions
            mask_data = np.zeros((height, width), dtype=np.uint8)
            
            # Random center for this object
            center_x = random.randint(width//4, 3*width//4)
            center_y = random.randint(height//4, 3*height//4)
            radius = random.randint(30, 100)
            
            # Create a simple circular mask
            y_indices, x_indices = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
            mask_data[dist_from_center <= radius] = 255
            
            mask.data = mask_data.tobytes()
            mask_images.append(mask)
            
            # Create a centroid point for this object
            centroid = Point()
            centroid.x = float(center_x)
            centroid.y = float(center_y)
            centroid.z = 0.0
            centroids.append(centroid)
        
        response.mask_images = mask_images
        response.centroids = centroids
        
        # Add a segmented image (just a copy of original with overlay)
        response.segmented_image = Image()
        response.segmented_image.height = height
        response.segmented_image.width = width
        response.segmented_image.encoding = "rgb8"
        response.segmented_image.data = np.ones((height, width, 3), dtype=np.uint8).tobytes()
        
        # Add contours (empty for simplicity)
        response.contours = []
        
        # Add header
        response.header = std_msgs.msg.Header()
        response.header.stamp = self.get_clock().now().to_msg()
        response.header.frame_id = "camera_color_optical_frame"
        
        self.get_logger().info(f"Created dummy SAM response with {num_objects} objects")
        return response

    def normal_to_quaternion(self, normal, align_axis='z'):
        """
        Convert a normal vector to a quaternion representing orientation.
        Ensures z-axis points downward relative to robot base frame.
        
        Args:
            normal: Normal vector (numpy array)
            align_axis: Which gripper axis to align with normal ('x', 'y', or 'z')
            
        Returns:
            quaternion as [x, y, z, w]
        """
        # Normalize the vector
        normal = np.array(normal, dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        
        # Ensure z-axis points downward (negative z in base frame)
        # If normal[2] is positive, flip the normal
        if normal[2] > 0:
            self.get_logger().info("Flipping normal to ensure downward orientation")
            normal = -normal
        
        # Choose which axis of the gripper aligns with the normal
        if align_axis == 'z':
            z_axis = normal
            
            # Find a reference vector not parallel to z_axis
            if abs(z_axis[0]) < abs(z_axis[1]) and abs(z_axis[0]) < abs(z_axis[2]):
                ref = np.array([1, 0, 0])
            elif abs(z_axis[1]) < abs(z_axis[2]):
                ref = np.array([0, 1, 0])
            else:
                ref = np.array([0, 0, 1])
            
            # Create orthogonal axes
            x_axis = np.cross(ref, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Form the rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Convert to quaternion
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        
        # For debugging, convert to RPY and log it
        rpy = self.quaternion_to_euler(quat)
        self.get_logger().info(f"Orientation RPY (degrees): roll={rpy[0]:.2f}°, pitch={rpy[1]:.2f}°, yaw={rpy[2]:.2f}°")
        
        return quat

    def quaternion_to_euler(self, quaternion):
        """
        Convert quaternion [x,y,z,w] to Euler angles [roll, pitch, yaw] in degrees.
        Uses ZYX rotation order (yaw, pitch, roll).
        """
        x, y, z, w = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]

    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x,y,z,w]."""
        trace = R[0,0] + R[1,1] + R[2,2]
        
        if trace > 0:
            S = 2.0 * np.sqrt(trace + 1.0)
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
        
        return [qx, qy, qz, qw]

    def q_yaw(self, theta):
        # Quaternion for rotation about local z-axis by θ:
        #    q = [x, y, z, w] = [0, 0, sin(θ/2), cos(θ/2)]
        return np.array([0.0, 0.0, np.sin(theta/2), np.cos(theta/2)])
    def quat_mul(self, q, r):
        x1,y1,z1,w1 = q
        x2,y2,z2,w2 = r
        # Hamilton product r ⊗ q (first q, then r)
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    def closest_flip_z(self, q):
        """
        Return the 180°-around-XY‐plane quaternion [ux,uy,0,0]
        that flips local Z → global –Z and is closest to q=[x,y,z,w].
        """
        x, y, z, w = q
        r = np.hypot(x, y)
        if r < 1e-8:
            # degenerate: choose X axis
            return np.array([1.0, 0.0, 0.0, 0.0])
        ux, uy = x / r, y / r
        return np.array([ux, uy, 0.0, 0.0])


def main(args=None):
    rclpy.init(args=args)
    
    # Make sure to import std_msgs for the Header
    import std_msgs.msg
    
    # Create the node
    service = RANSACSegmentationService()
    
    # Use a multithreaded executor to enable concurrent callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(service)
    
    try:
        service.get_logger().info('Starting multithreaded executor...')
        executor.spin()
    except KeyboardInterrupt:
        service.get_logger().info("Service interrupted by user")
    except Exception as e:
        service.get_logger().error(f"Error during service execution: {e}")
        import traceback
        service.get_logger().error(traceback.format_exc())
    finally:
        # Clean up
        service.get_logger().info("Shutting down service...")
        # Close any open OpenCV windows
        cv2.destroyAllWindows()
        service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()