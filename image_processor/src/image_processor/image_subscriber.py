#!/home/chrishj/ros2_env/bin/python3
import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import cv2
import numpy as np
import struct
import sensor_msgs_py.point_cloud2 as pc2
import math
from tf_transformations import quaternion_from_matrix
from ultralytics import YOLOE
import random
# Add TF2 imports
import tf2_ros
import geometry_msgs.msg
from tf2_geometry_msgs import do_transform_point
from rclpy.duration import Duration

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        
        # RGB Image subscription
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
            
        # PointCloud subscription
        # self.pointcloud_subscription = self.create_subscription(
        #     PointCloud2,
        #     '/camera/camera/depth/color/points',
        #     self.pointcloud_callback,
        #     10)
        
        self.depth_subscription = self.create_subscription(
        Image,
        '/camera/camera/aligned_depth_to_color/image_raw',
        self.depth_callback,
        10)

        self.fx = 641.415771484375  # Focal length x
        self.fy = 640.7596435546875 # Focal length y
        self.cx = 650.3182983398438# Principal point x
        self.cy = 357.72979736328125 # Principal point y
    
            
        # Set up TF2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.model = YOLOE("yoloe-11s-seg.pt")
        self.objects = ["package", "box"]
        self.model.set_classes(self.objects, self.model.get_text_pe(self.objects))  
        self.cv_image = None
        self.point_cloud_data = None
        
        # Image resolutions
        self.rgb_width, self.rgb_height = 1280, 720
        self.depth_width, self.depth_height = 848, 480
        
        
        # Target pixel in RGB image
        self.target_rgb_x, self.target_rgb_y = 755, 271
        
        # Calculate the corresponding pixel in depth image
        self.target_depth_x = int((self.target_rgb_x / self.rgb_width) * self.depth_width)
        self.target_depth_y = int((self.target_rgb_y / self.rgb_height) * self.depth_height)
        
        self.get_logger().info(f'RGB target: ({self.target_rgb_x}, {self.target_rgb_y})')
        self.get_logger().info(f'Depth target: ({self.target_depth_x}, {self.target_depth_y})')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image without CvBridge
            # First, get the data from the ROS Image message
            if msg.encoding != 'rgb8':
                self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
                return
                
            # Create a numpy array from the image data
            # For RGB8 format, each pixel is 3 bytes
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            # Reshape the array to the image dimensions
            # ROS uses row-major ordering
            image_data = image_data.reshape((msg.height, msg.width, 3))
            
            # Convert from RGB to BGR which is what OpenCV expects
            self.cv_image = image_data[..., ::-1].copy()  # ::-1 reverses the color channels
            
            # Draw the target point on the RGB image
            if self.cv_image is not None:
                img_with_point = self.cv_image.copy()
                results = self.model.predict(img_with_point)
                
                # Since results is a list, let's look at the first element
                if len(results) > 0:
                    result = results[0]  # Get the first result (for the single image we processed)
                    
                    # Check if names are available (class names)
                    if hasattr(result, 'names'):
                        pass
                        #self.get_logger().info(f"Class names: {result.names}")
                    
                    # Check if boxes are available
                    if hasattr(result, 'boxes'):
                        boxes_data = result.boxes
                        num_detections = len(boxes_data)
                        #self.get_logger().info(f"Number of detections: {num_detections}")
                        
                        # Draw each box
                        if hasattr(boxes_data, 'xyxy') and len(boxes_data.xyxy) > 0:
                            for i in range(len(boxes_data.xyxy)):
                                # Get box coordinates
                                box = boxes_data.xyxy[i].cpu().numpy() if hasattr(boxes_data.xyxy[i], 'cpu') else boxes_data.xyxy[i]
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Get class index and name
                                class_idx = int(boxes_data.cls[i].item() if hasattr(boxes_data.cls[i], 'item') else boxes_data.cls[i])
                                class_name = result.names[class_idx]
                                
                                # Get confidence
                                conf = float(boxes_data.conf[i].item() if hasattr(boxes_data.conf[i], 'item') else boxes_data.conf[i])
                                
                                # Draw bounding box
                                cv2.rectangle(img_with_point, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Draw label
                                label = f"{class_name} {conf:.2f}"
                                (label_width, label_height), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                                )
                                # Draw label background
                                cv2.rectangle(
                                    img_with_point, 
                                    (x1, y1 - label_height - 10), 
                                    (x1 + label_width, y1), 
                                    (0, 255, 0), 
                                    -1
                                )
                                # Draw label text
                                cv2.putText(
                                    img_with_point,
                                    label,
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    2
                                )
                                
                                # Calculate center point of detection
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                
                                # Draw center point
                                cv2.circle(img_with_point, (center_x, center_y), 4, (255, 0, 0), -1)
                                
                                # Try to get 3D point if we have point cloud data
                                
                                    
                                    # Queue this point for 3D position calculation
                                    # We'll do it in the next frame to avoid computation in this loop
                                    #self.target_depth_x = depth_x
                                    #self.target_depth_y = depth_y
                                
                                #self.get_logger().info(
                                #    f"Detection {i}: {class_name} (conf: {conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]"
                                #)
                    
                    # Check for segmentation masks and visualize if available
                    if hasattr(result, 'masks') and result.masks is not None:
                        #self.get_logger().info(f"Masks attributes: {dir(result.masks)}")
                        
                        if hasattr(result.masks, 'data') and result.masks.data is not None:
                            masks = result.masks.data
                            #self.get_logger().info(f"Masks shape: {masks.shape}")
                            
                            # Draw all masks with semi-transparency
                            for i, mask in enumerate(masks):
                                # Convert mask to numpy array if needed
                                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                                
                                # Resize mask to match input image dimensions
                                mask_resized = cv2.resize(
                                    mask_np.astype(np.uint8), 
                                    (img_with_point.shape[1], img_with_point.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                                
                                # Create a color mask image
                                color_mask = np.zeros_like(img_with_point, dtype=np.uint8)
                                
                                # Get class index for this mask
                                class_idx = int(result.boxes.cls[i].item() if hasattr(result.boxes.cls[i], 'item') else result.boxes.cls[i])
                                
                                # Choose color based on class
                                if class_idx == 0:  # package
                                    color = (0, 200, 255)  # orange-yellow
                                else:  # box
                                    color = (255, 0, 255)  # magenta
                                
                                # Set color for mask
                                mask_area = (mask_resized == 1)
                                color_mask[mask_area] = color
                                
                                # Blend with original image
                                alpha = 0.4  # Transparency factor
                                img_with_point[mask_area] = cv2.addWeighted(
                                    img_with_point[mask_area],
                                    1 - alpha,
                                    color_mask[mask_area],
                                    alpha,
                                    0
                                )
                                
                                # Annotate mask with class name
                                cv2.putText(
                                    img_with_point,
                                    f"Mask: {class_idx}",
                                    (10, 30 + i*30),  # Position text in top-left corner
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    color,
                                    2
                                )
                
                # Draw target point at the end so it's always visible
                cv2.circle(img_with_point, (self.target_rgb_x, self.target_rgb_y), 5, (0, 0, 255), -1)
                pixel_coords = [(845, 225), (880, 234), (882, 288), (845, 285)]
            
                # Draw each point with a unique color
                for i, (x, y) in enumerate(pixel_coords):
                    # Use cyan color (0, 255, 255) for the points
                    cv2.circle(img_with_point, (x, y), 5, (0, 255, 255), -1)
                    # Label each point
                    cv2.putText(
                        img_with_point,
                        str(i+1),  # Label points 1, 2, 3, 4
                        (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )
                
                # Connect the points to show the plane outline
                pts = np.array(pixel_coords, dtype=np.int32)
                cv2.polylines(img_with_point, [pts], True, (0, 255, 255), 2)
                
                # Display image with all visualizations
                cv2.imshow('Object Detection', img_with_point)
                cv2.waitKey(1)
                       
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return

    def depth_callback(self, msg):
        try:
            # Check the encoding
            if msg.encoding != '16UC1':
                self.get_logger().error(f'Unsupported depth encoding: {msg.encoding}')
                return
                
            # Create a numpy array from the depth data
            depth_data = np.frombuffer(msg.data, dtype=np.uint16)
            
            # Reshape the array to the image dimensions
            self.depth_image = depth_data.reshape((msg.height, msg.width))
            
            self.get_logger().debug('Received depth image')
            
            # Process frame if we have both RGB and depth
            if self.cv_image is not None and self.depth_image is not None:
                # Get plane information for custom box using RGB coordinates directly
                pixel_coords = [(788, 206), (877, 196), (881, 284), (804, 292)]
                x_min = min([coord[0] for coord in pixel_coords])
                x_max = max([coord[0] for coord in pixel_coords])
                y_min = min([coord[1] for coord in pixel_coords])
                y_max = max([coord[1] for coord in pixel_coords])

                points_3d = self.get_points_from_depth_region(x_min, x_max, y_min, y_max)

                plane_info = self.get_plane_info_ransac(points_3d, iterations=100, threshold=0.01)

                gripper_quaternion = self.normal_to_approach_quaternion(plane_info["normal"])
                self.get_logger().info(f"Gripper Quaternion: {gripper_quaternion}")


                # for x in range(x_min, x_max):
                #     for y in range(y_min, y_max):
                #         point_test = self.get_point_from_depth(x, y)
                #plane_info = self.get_plane_info(pixel_coords)
                
                # If we have plane info, transform it to robot coordinates
                #if plane_info is not None:
                        # Calculate TCP pose for approaching this plane
                    # tcp_pose = self.get_tcp_pose_for_plane(plane_info)
                    
                    # if tcp_pose:
                    #     pass
                    #     self.get_logger().info("========== TCP POSE FOR PLANE APPROACH ==========")
                    #     self.get_logger().info(f"Position: [{tcp_pose['position'][0]:.4f}, {tcp_pose['position'][1]:.4f}, {tcp_pose['position'][2]:.4f}]")
                    #     self.get_logger().info(f"Orientation: [{tcp_pose['orientation'][0]:.4f}, {tcp_pose['orientation'][1]:.4f}, {tcp_pose['orientation'][2]:.4f}, {tcp_pose['orientation'][3]:.4f}]")
                    #     self.get_logger().info("=============================================")
                    
                    # # Get current TCP pose 
                    # tcp_position, tcp_orientation = self.get_current_tcp_pose()
                    
                    # if tcp_position is not None:
                    #     # Calculate movement needed for TCP
                    #     movement = self.calculate_tcp_movement(
                    #         plane_info["center"],
                    #         plane_info["normal"],
                    #         tcp_position
                    #     )
                    
        except Exception as e:
            self.get_logger().error(f'Depth image processing error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def get_points_from_depth_region(self, min_x, max_x, min_y, max_y):
        """
        Get 3D points from depth image for all pixels within a rectangular region.
        
        Args:
            min_x, max_x, min_y, max_y: Integer bounds defining the rectangular region
            
        Returns:
            NumPy array of 3D points in robot base frame with shape (n, 3)
            where n is the number of valid points in the region
        """
        if self.depth_image is None:
            self.get_logger().warn("No depth image available")
            return None
        
        height, width = self.depth_image.shape
        
        # Ensure coordinates are within image bounds
        min_x = max(0, min_x)
        max_x = min(width - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(height - 1, max_y)
        
        # Check if the region is valid
        if min_x > max_x or min_y > max_y:
            self.get_logger().error(f"Invalid region: [{min_x}, {max_x}] x [{min_y}, {max_y}]")
            return np.array([])
        
        # Create meshgrid of all (x,y) coordinates in the region
        y_indices, x_indices = np.mgrid[min_y:max_y+1, min_x:max_x+1]
        x_indices = x_indices.flatten()
        y_indices = y_indices.flatten()
        
        # Get all depth values at once
        depth_values = self.depth_image[y_indices, x_indices]
        
        # Filter out invalid depth values (zeros)
        valid_depth = depth_values > 0
        valid_x = x_indices[valid_depth]
        valid_y = y_indices[valid_depth]
        valid_depths = depth_values[valid_depth]
        
        if not np.any(valid_depth):
            self.get_logger().warn("No valid depth readings found in the specified region")
            return np.array([])
        
        # Convert depths from mm to meters
        depths_m = valid_depths / 1000.0
        
        # Calculate 3D points using camera model (vectorized)
        X = (valid_x - self.cx) * depths_m / self.fx
        Y = (valid_y - self.cy) * depths_m / self.fy
        Z = depths_m
        
        # Combine into array of camera frame points
        points_3d_camera = np.column_stack((X, Y, Z))
        
        # Log summary information
        num_total_pixels = len(x_indices)
        num_valid_pixels = len(valid_x)
        self.get_logger().info(
            f"Region [{min_x}, {max_x}] x [{min_y}, {max_y}]: "
            f"Found {num_valid_pixels} valid points out of {num_total_pixels} pixels"
        )
        
        try:
            # Transform all valid points to robot base frame
            np.save("points_3d_camera.npy", points_3d_camera)
            points_3d_robot = self.transform_points_to_robot_base_vectorized(points_3d_camera)
            
            # Log summary of transformed points
            self.get_logger().info(f"Successfully transformed {len(points_3d_robot)} points to robot base frame")
            
            # Optionally, log some statistical information about the points
            if len(points_3d_robot) > 0:
                min_values = np.min(points_3d_robot, axis=0)
                max_values = np.max(points_3d_robot, axis=0)
                mean_values = np.mean(points_3d_robot, axis=0)
                
                self.get_logger().info(
                    f"Point cloud statistics in robot base frame:\n"
                    f"  Min: X={min_values[0]:.3f}, Y={min_values[1]:.3f}, Z={min_values[2]:.3f}\n"
                    f"  Max: X={max_values[0]:.3f}, Y={max_values[1]:.3f}, Z={max_values[2]:.3f}\n"
                    f"  Mean: X={mean_values[0]:.3f}, Y={mean_values[1]:.3f}, Z={mean_values[2]:.3f}"
                )
            
            return points_3d_robot
        
        except Exception as e:
            self.get_logger().error(f'Error in point transformation: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return np.array([])
        
    def transform_points_to_robot_base_vectorized(self, points_3d):
        """
        Transform multiple points from camera_color_optical_frame to the robot's base frame (ur10_base_link)
        using a static transformation matrix with coordinate frame adjustment.
        
        Args:
            points_3d: List of numpy arrays or a 2D numpy array with shape (n, 3)
                    where each row is [x, y, z] in camera_color_optical_frame
        Returns:
            Numpy array of transformed points in robot base frame (ur10_base_link)
        """
        # Static transformation matrix from camera_color_optical_frame to ur10_base_link
        transform_matrix = np.array([
            [0.995, 0.095, 0.007, -0.510],
            [0.095, -0.995, -0.005, 0.010],
            [0.006, 0.006, -1.000, 1.110],
            [0.000, 0.000, 0.000, 1.000]
        ])
        
        # Coordinate convention adjustment matrix based on our analysis
        # Camera X → Robot -X, Camera Y → Robot -Y, Camera Z → Robot Z
        coordinate_adjustment = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Combine the transformations
        # First apply coordinate adjustment, then apply the TF matrix
        combined_transform = transform_matrix @ coordinate_adjustment
        
        # Convert list to numpy array if needed
        if isinstance(points_3d, list):
            points_3d = np.array(points_3d)
            
        # Make sure points are in the right shape
        if len(points_3d.shape) == 1:
            # Single point, reshape to (1, 3)
            points_3d = points_3d.reshape(1, 3)
            
        # Add homogeneous coordinate (column of ones)
        n_points = points_3d.shape[0]
        points_homogeneous = np.hstack((points_3d, np.ones((n_points, 1))))
        
        # Apply combined transformation to all points at once
        transformed_points_homogeneous = np.dot(points_homogeneous, combined_transform.T)
        
        # Log information if needed
        if n_points > 0:
            self.get_logger().info(f"First point before transform: {points_3d[0]}")
            self.get_logger().info(f"First point after transform: {transformed_points_homogeneous[0]}")
            
            # Test with origin (should match the translation component of the combined matrix)
            origin_test = np.array([[0, 0, 0, 1]])
            origin_transformed = np.dot(origin_test, combined_transform.T)
            self.get_logger().info(f"Origin (0,0,0) transforms to: {origin_transformed[0, :3]}")

        transformed_points_homogeneous[:, 0] = -transformed_points_homogeneous[:, 0]  # Flip X sign
        transformed_points_homogeneous[:, 1] = -transformed_points_homogeneous[:, 1]  # Flip Y sign
        # Return just the x, y, z coordinates
        return transformed_points_homogeneous[:, :3]
    
    def get_plane_info_ransac(self, points_3d, iterations=100, threshold=0.01):
        """
        Calculate the normal vector and center point of the best-fit plane
        using the RANSAC algorithm. Assumes points are already in the desired frame (e.g., robot base).

        Args:
            points_3d: NumPy array with shape (n, 3) representing 3D points.
            iterations: Number of RANSAC iterations to perform.
            threshold: Maximum distance for a point to be considered an inlier (in meters).

        Returns:
            dict: Contains 'normal', 'center', and 'inliers' (NumPy array of points)
                for the best-fit plane, or None if no suitable plane is found.
        """
        # Convert to numpy array if input is a list
        if isinstance(points_3d, list):
            points_array = np.array(points_3d)
        else:
            points_array = points_3d
        
        num_points = len(points_array)
        
        if num_points < 3:
            self.get_logger().warn(f"RANSAC needs at least 3 points, got {num_points}. Skipping plane fitting.")
            return None

        best_inliers_count = -1
        best_plane_params = None
        best_inlier_indices = None

        for i in range(iterations):
            # 1. Randomly select 3 unique points
            try:
                sample_indices = np.random.choice(num_points, 3, replace=False)
                sample_points = points_array[sample_indices]
            except ValueError:
                self.get_logger().error(f"Could not sample 3 unique points from {num_points} available points.")
                return None

            p1, p2, p3 = sample_points[0], sample_points[1], sample_points[2]

            # 2. Fit a plane model (ax + by + cz + d = 0)
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)

            # Check for collinear points (normal vector is zero length)
            if normal_norm < 1e-6:
                continue # Try next iteration

            normal = normal / normal_norm # Normalize the normal vector

            # Calculate d using one of the points (p1)
            d = -np.dot(normal, p1)

            # 3. Count inliers - vectorized distance calculation
            distances = np.abs(np.dot(points_array, normal) + d)
            current_inlier_indices = np.where(distances < threshold)[0]
            current_inliers_count = len(current_inlier_indices)

            # 4. Update best model if current one is better
            if current_inliers_count > best_inliers_count:
                best_inliers_count = current_inliers_count
                best_plane_params = (normal, d) # Store normal and d
                best_inlier_indices = current_inlier_indices
                # Optional: Early exit if we find a good model
                # if best_inliers_count > num_points * 0.9:
                #     break

        # 5. After iterations, check if a model was found
        if best_plane_params is None or best_inliers_count < 3: # Need at least 3 inliers
            self.get_logger().warn(f"RANSAC failed to find a suitable plane model with enough inliers ({best_inliers_count} found).")
            return None

        # Retrieve the best normal and the corresponding inlier points
        final_normal, final_d = best_plane_params
        final_inlier_points = points_array[best_inlier_indices]

        # Calculate center as mean of inlier points
        center = np.mean(final_inlier_points, axis=0)

        # Optional: Refinement step - refit plane using all inliers
        # You could add SVD-based plane fitting here for better accuracy
        
        result = {
            "normal": final_normal,
            "center": center,
            "inliers": final_inlier_points,  # Keep as NumPy array
            "inlier_indices": best_inlier_indices,  # New: return the indices of inliers for reference
            "inlier_count": best_inliers_count,  # New: return the count directly
            "total_points": num_points  # New: return the total number of points
        }

        self.get_logger().info(f"RANSAC found plane with {best_inliers_count} inliers out of {num_points} points.")
        
        try:
            frame_id = self.tf_buffer.lookup_transform('ur10_base_link', 'camera_color_optical_frame', 
                                                    rclpy.time.Time()).header.frame_id
            self.get_logger().info(f"Plane Normal (in {frame_id}): {result['normal']}")
            self.get_logger().info(f"Plane Center (in {frame_id}): {result['center']}")
        except Exception as e:
            # If tf lookup fails, just log without the frame information
            self.get_logger().info(f"Plane Normal: {result['normal']}")
            self.get_logger().info(f"Plane Center: {result['center']}")

        return result

    def normal_to_approach_quaternion(self, normal_vector, approach_direction='negative'):
        """
        Convert a plane normal to a quaternion with Y as the dominant component.
        Preserves the natural quaternion values while ensuring Y dominance.
        
        Args:
            normal_vector: NumPy array [nx, ny, nz] representing the plane normal
            approach_direction: 'negative' to approach from above the plane (default),
                            'positive' to approach from below
        
        Returns:
            Quaternion as [qx, qy, qz, qw] for gripper orientation in ur10_base_link frame
        """
        # Normalize the normal vector
        normal = normal_vector / np.linalg.norm(normal_vector)
        
        # Force all normals to point upward for consistency
        if normal[2] < 0:
            normal = -normal
        
        # For approaching from above, want the gripper's z-axis to point opposite to the normal
        if approach_direction == 'negative':
            z_axis = -normal
        else:
            z_axis = normal
        
        # Always use world y-axis as reference when possible
        world_y = np.array([0, 1, 0])
        
        # Check if z_axis is too close to world_y
        if np.abs(np.dot(z_axis, world_y)) > 0.9:
            reference = np.array([1, 0, 0])
        else:
            reference = world_y
        
        # Create orthogonal axes
        x_axis = np.cross(reference, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Build rotation matrix
        R = np.column_stack((x_axis, y_axis, z_axis))
        
        # Convert rotation matrix to quaternion
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        
        # Get absolute values to compare magnitudes
        abs_components = np.abs([qx, qy, qz, qw])
        
        # Check if Y is already dominant
        if abs_components[1] >= np.max(abs_components):
            # Y is already dominant, ensure it's positive
            if qy < 0:
                return np.array([-qx, -qy, -qz, -qw])
            return np.array([qx, qy, qz, qw])
        
        # Y is not dominant, need to transform the quaternion
        # First, find which component is dominant
        dominant_idx = np.argmax(abs_components)
        
        # Create a rotation quaternion to transform the dominant component to y
        # These are 90° rotations to move x→y, z→y, or w→y
        transform_quaternions = [
            [0, 0, -0.7071, 0.7071],  # x→y: 90° rotation around z
            [0, 0, 0, 0],             # y→y: no rotation (placeholder)
            [0.7071, 0, 0, 0.7071],   # z→y: 90° rotation around x
            [0, 0.7071, 0, 0.7071]    # w→y: 90° rotation around y (for completeness)
        ]
        
        # Apply the appropriate rotation to make y dominant
        if dominant_idx != 1:  # If y is not already dominant
            rotation = transform_quaternions[dominant_idx]
            
            # Quaternion multiplication to apply rotation
            result = self.quaternion_multiply(rotation, [qx, qy, qz, qw])
            
            # Normalize result
            result = result / np.linalg.norm(result)
            
            # Ensure y is positive
            if result[1] < 0:
                result = -result
                
            return result
        
        # Should never reach here
        return np.array([qx, qy, qz, qw])

    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions [x,y,z,w]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        
        return np.array([x, y, z, w])

    # def pointcloud_callback(self, msg):
    #     try:
    #         # Print field information from the first message
    #         if not hasattr(self, 'fields_printed'):
    #             self.get_logger().warn(f"PointCloud2 fields: {[field.name for field in msg.fields]}")
    #             self.get_logger().warn(f"PointCloud2 dimensions: width={msg.width}, height={msg.height}")
    #             self.fields_printed = True
            
    #         # Store the point cloud data
    #         self.point_cloud_data = msg
            
    #         # Extract 3D coordinates for the target pixel
    #         if self.point_cloud_data is not None:
    #             #center_box_3d = self.get_point_from_cloud(self.target_depth_x, self.target_depth_y)
                


    #             # Get plane information for custom box
    #             #pixel_coords = [(731, 249), (727, 286), (780, 290), (781, 240)]
    #             pixel_coords = [(885, 225), (920, 234), (922, 288), (885, 285)]
    #             plane_info = self.get_plane_info(pixel_coords)
                
    #             # If we have plane info, transform it to robot coordinates
    #             if plane_info is not None:
    #                 # Transform center point to robot frame
    #                 robot_center = self.transform_point_to_robot_base(plane_info["center"])
                    
    #                 # Transform a point along the normal to correctly get the normal in robot frame
    #                 normal_point = plane_info["center"] - plane_info["normal"] * 0.1  # 10cm along normal
    #                 robot_normal_point = self.transform_point_to_robot_base(normal_point)
                    
    #                 if robot_center is not None and robot_normal_point is not None:
    #                     # Calculate the new normal in robot frame
    #                     robot_normal = robot_normal_point - robot_center
    #                     robot_normal_length = np.linalg.norm(robot_normal)
    #                     if robot_normal_length > 0:
    #                         robot_normal = robot_normal / robot_normal_length
                        
    #                     self.get_logger().info(f"Robot frame normal: {robot_normal}")
    #                     self.get_logger().info(f"Robot frame center: {robot_center}")
                        
    #                     # Update plane_info with robot frame data
    #                     plane_info["robot_normal"] = robot_normal
    #                     plane_info["robot_center"] = robot_center
                        
    #                     # Calculate TCP pose for approaching this plane
    #                     tcp_pose = self.get_tcp_pose_for_plane(plane_info)
                        
    #                     if tcp_pose:
    #                         self.get_logger().info("========== TCP POSE FOR PLANE APPROACH ==========")
    #                         self.get_logger().info(f"Position: [{tcp_pose['position'][0]:.4f}, {tcp_pose['position'][1]:.4f}, {tcp_pose['position'][2]:.4f}]")
    #                         self.get_logger().info(f"Orientation: [{tcp_pose['orientation'][0]:.4f}, {tcp_pose['orientation'][1]:.4f}, {tcp_pose['orientation'][2]:.4f}, {tcp_pose['orientation'][3]:.4f}]")
    #                         self.get_logger().info("=============================================")
                        
    #                     # Get current TCP pose 
    #                     tcp_position, tcp_orientation = self.get_current_tcp_pose()
                        
    #                     if tcp_position is not None:
    #                         # Calculate movement needed for TCP
    #                         movement = self.calculate_tcp_movement(
    #                             plane_info["robot_center"],
    #                             plane_info["robot_normal"],
    #                             tcp_position
    #                         )
                
    #     except Exception as e:
    #         self.get_logger().error(f'PointCloud processing error: {e}')
    #         import traceback
    #         self.get_logger().error(traceback.format_exc())

    def transform_point_to_robot_base(self, point_3d):
        """
        Transform a point from camera_depth_optical_frame to the robot's base frame.
        
        Args:
            point_3d: numpy array [x, y, z] in camera_depth_optical_frame
            
        Returns:
            Transformed point in robot base frame
        """
        # Create a PointStamped message
        point_stamped = geometry_msgs.msg.PointStamped()
        point_stamped.header.frame_id = "camera_color_optical_frame"
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.point.x = float(point_3d[0])
        point_stamped.point.y = float(point_3d[1])
        point_stamped.point.z = float(point_3d[2])
        
        try:
            # Look up transform from camera frame to robot base
            transform = self.tf_buffer.lookup_transform(
                "ur10_base_link",  # Target frame - your UR10 base
                "camera_color_optical_frame",  # Source frame
                rclpy.time.Time(),  # Get latest transform
                Duration(seconds=0.05)  # Wait for up to 1 second
            )
            
            # Apply transform
            transformed_point = do_transform_point(point_stamped, transform)
            
            # self.get_logger().info(
            #     f"Transformed point: X={transformed_point.point.x:.3f}, "
            #     f"Y={transformed_point.point.y:.3f}, Z={transformed_point.point.z:.3f}"
            # )
            
            return np.array([
                transformed_point.point.x,
                transformed_point.point.y,
                transformed_point.point.z
            ])
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF error: {e}')
            return None
            
    def get_current_tcp_pose(self):
        try:
            # Look up transform from base to TCP
            transform = self.tf_buffer.lookup_transform(
                "ur10_base_link",  # Target frame - your UR10 base
                "ur10_tcp",        # Source frame - your UR10 TCP
                rclpy.time.Time()
            )
            
            # Extract position
            position = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            
            # Extract rotation (as quaternion)
            orientation = [
                transform.transform.rotation.x,
                transform.transform.rotation.y, 
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            
            self.get_logger().info(f"Current TCP position: {position}")
            return position, orientation
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF error getting TCP pose: {e}')
            return None, None
            
    def calculate_tcp_movement(self, robot_center, robot_normal, current_tcp_pose):
        """
        Calculate how much to move the TCP to reach the target point along the normal.
        
        Args:
            robot_center: Center point in robot base frame
            robot_normal: Normal vector in robot base frame
            current_tcp_pose: Current TCP position [x, y, z]
            
        Returns:
            Movement vector
        """
        # Convert current_tcp_pose to numpy array if it's not already
        current_tcp_pos = np.array(current_tcp_pose)
        
        # Distance from current TCP to center point
        tcp_to_center = robot_center - current_tcp_pos
        
        # Project this distance onto the normal vector
        distance_along_normal = np.dot(tcp_to_center, robot_normal)
        
        # The movement vector is this distance along the normal direction
        movement = distance_along_normal * robot_normal
        
        self.get_logger().info(f"TCP needs to move: {movement}")
        self.get_logger().info(f"Distance to move down: {distance_along_normal:.4f} meters")
        
        return movement
    
    def get_plane_info(self, pixel_coords):
        """
        Calculate the normal vector and center point of a plane defined by 3D points
        corresponding to the given pixel coordinates.
        
        Args:
            pixel_coords: List of tuples [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        
        Returns:
            dict: Contains normal vector, center point, and whether points are planar
        """
        if self.depth_image is None:
            self.get_logger().warn("No depth image available")
            return None
        

        points_3d = []
        for x, y in pixel_coords:
            point = self.get_point_from_depth(x, y)
            
            if point is not None:
                points_3d.append(point)
                #self.get_logger().info(f'3D point at pixel ({x}, {y}): X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f} meters')
            else:
                self.get_logger().warn(f'Could not get 3D point for pixel ({x}, {y})')
                return None
        
        if len(points_3d) != 4:
            self.get_logger().warn(f"Expected 4 valid 3D points, got {len(points_3d)}")
            return None

        # Calculate normal vector
        v1 = points_3d[1] - points_3d[0]
        v2 = points_3d[2] - points_3d[0]
        normal = np.cross(v1, v2)
        
        # Normalize the normal vector
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = normal / normal_length
        
        # Calculate center point
        center = np.mean(points_3d, axis=0)
        
        # Check if fourth point is on the plane
        v3 = points_3d[3] - points_3d[0]
        dot_product = np.dot(normal, v3)
        is_planar = np.isclose(dot_product, 0, atol=1e-3)  # Use slightly larger tolerance for real-world data
        
        result = {
            "normal": normal,
            "center": center,
            "is_planar": is_planar,
            "points_3d": points_3d
        }
        
        # self.get_logger().info(f"Plane normal: {normal}")
        # self.get_logger().info(f"Plane center: {center}")
        # self.get_logger().info(f"Is planar: {is_planar}")
        
        return result

    def get_plane_info_ransac(self, points_3d, iterations=100, threshold=0.01):
        """
        Calculate the normal vector and center point of the best-fit plane
        using the RANSAC algorithm.

        Args:
            points_3d: List of numpy arrays, each representing a 3D point [x, y, z].
            iterations: Number of RANSAC iterations to perform.
            threshold: Maximum distance for a point to be considered an inlier.

        Returns:
            dict: Contains 'normal', 'center', and 'inliers' (list of points)
                for the best-fit plane, or None if no suitable plane is found.
        """
        if len(points_3d) < 3:
            # Not enough points to define a plane
            # self.get_logger().warn("RANSAC needs at least 3 points.") # Add logger if used inside the class
            print("RANSAC needs at least 3 points.") # Standalone print
            return None

        best_inliers_count = -1
        best_plane_params = None
        best_inlier_indices = None

        num_points = len(points_3d)
        points_array = np.array(points_3d) # Convert list of arrays to a single 2D numpy array

        for i in range(iterations):
            # 1. Randomly select 3 unique points
            try:
                sample_indices = random.sample(range(num_points), 3)
                sample_points = points_array[sample_indices]
            except ValueError:
                # Handle cases where num_points < 3, though checked earlier
                # self.get_logger().error("Could not sample 3 points.")
                print("Could not sample 3 points.")
                return None

            p1, p2, p3 = sample_points[0], sample_points[1], sample_points[2]

            # 2. Fit a plane model (ax + by + cz + d = 0)
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)

            # Check for collinear points (normal vector is zero length)
            if normal_norm < 1e-6:
                continue # Try next iteration

            normal = normal / normal_norm # Normalize the normal vector

            # Calculate d using one of the points (p1)
            d = -np.dot(normal, p1)

            # 3. Count inliers
            current_inlier_indices = []
            # Calculate distances from all points to the plane: |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
            # Since normal is normalized, sqrt(a^2+b^2+c^2) = 1
            distances = np.abs(np.dot(points_array, normal) + d)

            current_inlier_indices = np.where(distances < threshold)[0]
            current_inliers_count = len(current_inlier_indices)

            # 4. Update best model if current one is better
            if current_inliers_count > best_inliers_count:
                best_inliers_count = current_inliers_count
                best_plane_params = (normal, d)
                best_inlier_indices = current_inlier_indices
                # Optional: Add an early exit condition if a certain percentage of points are inliers

        # 5. After iterations, use the best model
        if best_plane_params is None:
            # self.get_logger().warn("RANSAC failed to find a suitable plane.")
            print("RANSAC failed to find a suitable plane.")
            return None

        # Optional: Refit the plane using all inliers from the best iteration for better accuracy
        final_normal, final_d = best_plane_params
        final_inlier_points = points_array[best_inlier_indices]

        # Use SVD for robust plane fitting on inliers if desired, or keep RANSAC normal
        # For simplicity, we'll use the RANSAC normal and calculate center from inliers
        center = np.mean(final_inlier_points, axis=0)

        result = {
            "normal": final_normal,
            "center": center,
            "inliers": final_inlier_points.tolist() # Convert back to list of arrays if needed
        }

        # self.get_logger().info(f"RANSAC found plane with {best_inliers_count} inliers.")
        # self.get_logger().info(f"Plane normal: {result['normal']}")
        # self.get_logger().info(f"Plane center: {result['center']}")
        print(f"RANSAC found plane with {best_inliers_count} inliers.") # Standalone print
        print(f"Plane normal: {result['normal']}")
        print(f"Plane center: {result['center']}")

        return result

    def get_point_from_cloud(self, x, y):
        if self.point_cloud_data is None:
            self.get_logger().warn("No point cloud data available")
            return
        
        # Debug: Print coordinates we're trying to access
        self.get_logger().info(f"Attempting to get point at depth coordinates: ({x}, {y})")
        
        # Ensure coordinates are within bounds
        max_x = self.point_cloud_data.width - 1
        max_y = self.point_cloud_data.height - 1
        
        if x < 0 or x > max_x or y < 0 or y > max_y:
            self.get_logger().error(f"Coordinates ({x}, {y}) are out of bounds for point cloud dimensions {self.point_cloud_data.width}x{self.point_cloud_data.height}")
            return
        
        try:
            # Get the actual field names from the point cloud
            actual_fields = [field.name for field in self.point_cloud_data.fields]
            self.get_logger().info(f"Actual fields in point cloud: {actual_fields}")
            
            # Calculate row-major index for the point
            point_index = y * self.point_cloud_data.width + x
            
            # Use read_points with row_step to get the specific point
            gen = pc2.read_points(self.point_cloud_data, 
                                field_names=actual_fields,
                                skip_nans=False,
                                uvs=None)  # Not using uvs parameter
                                
            # Convert generator to list so we can index it
            points_list = list(gen)
            
            if point_index < len(points_list):
                point = points_list[point_index]
                
                # Log the entire point for debugging
                self.get_logger().info(f"Raw point data: {point}")
                
                # Extract XYZ coordinates
                if len(point) >= 3 and not (math.isnan(point[0]) or math.isnan(point[1]) or math.isnan(point[2])):
                    self.get_logger().info(f'3D point at pixel ({x}, {y}): X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f} meters')
                    self.get_logger().info(f'3D center point: {point}')
                    self.transform_point_to_robot_base(np.array([float(point[0]), float(point[1]), float(point[2])]))
                else:
                    self.get_logger().warn(f'Point contains NaN values or insufficient data: {point}')
            else:
                self.get_logger().warn(f'Point index {point_index} is out of range for {len(points_list)} points')
                    
        except Exception as e:
            self.get_logger().error(f'Error extracting point: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def calculate_tcp_orientation_from_normal(self, normal_vector):
        """
        Calculate TCP orientation quaternion from a plane normal vector.
        The gripper Z-axis should point opposite to the normal vector for proper approach.
        
        Args:
            normal_vector: Normal vector of the plane in robot base frame
            
        Returns:
            Quaternion [x, y, z, w] representing the orientation
        """
        
        
        # The Z-axis of the TCP should point opposite to the normal vector
        z_axis = -normal_vector  # Negative because gripper approaches opposite to normal
        
        # Find a vector not parallel to z_axis to create coordinate system
        if abs(np.dot(z_axis, [0, 0, 1])) > 0.9:
            reference = np.array([1, 0, 0])  # Use x-axis if z_axis is close to world z
        else:
            reference = np.array([0, 0, 1])  # Otherwise use world z
        
        # Calculate y-axis as cross product of z-axis and reference
        y_axis = np.cross(z_axis, reference)
        y_axis = y_axis / np.linalg.norm(y_axis)  # Normalize
        
        # Calculate x-axis as cross product of y-axis and z-axis
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)  # Normalize
        
        # Create rotation matrix from the three axes
        rotation_matrix = np.zeros((4, 4))
        rotation_matrix[0:3, 0] = x_axis
        rotation_matrix[0:3, 1] = y_axis
        rotation_matrix[0:3, 2] = z_axis
        rotation_matrix[3, 3] = 1.0
        
        # Convert rotation matrix to quaternion
        quaternion = quaternion_from_matrix(rotation_matrix)
        
        return quaternion
    
    def get_tcp_pose_for_plane(self, plane_info):
        """
        Calculate TCP pose (position and orientation) for approaching the plane.
        
        Args:
            plane_info: Dictionary with plane center and normal
            
        Returns:
            Dictionary with position and orientation for TCP
        """
        if "center" not in plane_info or "normal" not in plane_info:
            self.get_logger().error("Plane information not available in robot frame")
            return None
            
        # Get position (already in robot frame)
        position = plane_info["center"]
        
        # Calculate orientation based on normal vector
        orientation = self.calculate_tcp_orientation_from_normal(plane_info["normal"])
        
        tcp_pose = {
            "position": position,
            "orientation": orientation
        }
        
        # Log the information
        #self.get_logger().info(f"TCP Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        #self.get_logger().info(f"TCP Orientation (quaternion): [{orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}, {orientation[3]:.4f}]")
        
        return tcp_pose

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()