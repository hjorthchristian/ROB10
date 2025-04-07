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
        self.pointcloud_subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10)
            
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
                                if self.point_cloud_data is not None:
                                    # Convert RGB center to depth coordinates
                                    depth_x = int((center_x / self.rgb_width) * self.depth_width)
                                    depth_y = int((center_y / self.rgb_height) * self.depth_height)
                                    
                                    # Add text to show we're computing 3D position
                                    cv2.putText(
                                        img_with_point,
                                        f"3D pos for: {depth_x},{depth_y}",
                                        (center_x, center_y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        (255, 0, 0),
                                        1
                                    )
                                    
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
                
                # Display image with all visualizations
                cv2.imshow('Object Detection', img_with_point)
                cv2.waitKey(1)
                       
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return

    def pointcloud_callback(self, msg):
        try:
            # Print field information from the first message
            if not hasattr(self, 'fields_printed'):
                self.get_logger().warn(f"PointCloud2 fields: {[field.name for field in msg.fields]}")
                self.get_logger().warn(f"PointCloud2 dimensions: width={msg.width}, height={msg.height}")
                self.fields_printed = True
            
            # Store the point cloud data
            self.point_cloud_data = msg
            
            # Extract 3D coordinates for the target pixel
            if self.point_cloud_data is not None:
                center_box_3d = self.get_point_from_cloud(self.target_depth_x, self.target_depth_y)
                center_box_3d


                # Get plane information for custom box
                pixel_coords = [(731, 249), (727, 286), (780, 290), (781, 240)]
                plane_info = self.get_plane_info(pixel_coords)
                
                # If we have plane info, transform it to robot coordinates
                if plane_info is not None:
                    # Transform center point to robot frame
                    robot_center = self.transform_point_to_robot_base(plane_info["center"])
                    
                    # Transform a point along the normal to correctly get the normal in robot frame
                    normal_point = plane_info["center"] - plane_info["normal"] * 0.1  # 10cm along normal
                    robot_normal_point = self.transform_point_to_robot_base(normal_point)
                    
                    if robot_center is not None and robot_normal_point is not None:
                        # Calculate the new normal in robot frame
                        robot_normal = robot_normal_point - robot_center
                        robot_normal_length = np.linalg.norm(robot_normal)
                        if robot_normal_length > 0:
                            robot_normal = robot_normal / robot_normal_length
                        
                        self.get_logger().info(f"Robot frame normal: {robot_normal}")
                        self.get_logger().info(f"Robot frame center: {robot_center}")
                        
                        # Update plane_info with robot frame data
                        plane_info["robot_normal"] = robot_normal
                        plane_info["robot_center"] = robot_center
                        
                        # Calculate TCP pose for approaching this plane
                        tcp_pose = self.get_tcp_pose_for_plane(plane_info)
                        
                        if tcp_pose:
                            self.get_logger().info("========== TCP POSE FOR PLANE APPROACH ==========")
                            self.get_logger().info(f"Position: [{tcp_pose['position'][0]:.4f}, {tcp_pose['position'][1]:.4f}, {tcp_pose['position'][2]:.4f}]")
                            self.get_logger().info(f"Orientation: [{tcp_pose['orientation'][0]:.4f}, {tcp_pose['orientation'][1]:.4f}, {tcp_pose['orientation'][2]:.4f}, {tcp_pose['orientation'][3]:.4f}]")
                            self.get_logger().info("=============================================")
                        
                        # Get current TCP pose 
                        tcp_position, tcp_orientation = self.get_current_tcp_pose()
                        
                        if tcp_position is not None:
                            # Calculate movement needed for TCP
                            movement = self.calculate_tcp_movement(
                                plane_info["robot_center"],
                                plane_info["robot_normal"],
                                tcp_position
                            )
                
        except Exception as e:
            self.get_logger().error(f'PointCloud processing error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

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
        point_stamped.header.frame_id = "camera_depth_optical_frame"
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.point.x = float(point_3d[0])
        point_stamped.point.y = float(point_3d[1])
        point_stamped.point.z = float(point_3d[2])
        
        try:
            # Look up transform from camera frame to robot base
            transform = self.tf_buffer.lookup_transform(
                "ur10_base_link",  # Target frame - your UR10 base
                "camera_depth_optical_frame",  # Source frame
                rclpy.time.Time(),  # Get latest transform
                Duration(seconds=1.0)  # Wait for up to 1 second
            )
            
            # Apply transform
            transformed_point = do_transform_point(point_stamped, transform)
            
            self.get_logger().info(
                f"Transformed point: X={transformed_point.point.x:.3f}, "
                f"Y={transformed_point.point.y:.3f}, Z={transformed_point.point.z:.3f}"
            )
            
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
        if self.point_cloud_data is None:
            self.get_logger().warn("No point cloud data available")
            return None
        
        # Convert RGB pixel coordinates to depth coordinates
        depth_coords = []
        for x, y in pixel_coords:
            depth_x = int((x / self.rgb_width) * self.depth_width)
            depth_y = int((y / self.rgb_height) * self.depth_height)
            depth_coords.append((depth_x, depth_y))
        
        # Get 3D points for each pixel
        points_3d = []
        for x, y in depth_coords:
            # Calculate row-major index for the point
            point_index = y * self.point_cloud_data.width + x
            
            # Get field names
            actual_fields = [field.name for field in self.point_cloud_data.fields]
            
            # Read all points
            points_list = list(pc2.read_points(
                self.point_cloud_data, 
                field_names=actual_fields,
                skip_nans=False
            ))
            
            if point_index < len(points_list):
                point = points_list[point_index]
                
                # Extract XYZ coordinates
                if len(point) >= 3 and not (math.isnan(point[0]) or math.isnan(point[1]) or math.isnan(point[2])):
                    points_3d.append(np.array([float(point[0]), float(point[1]), float(point[2])]))  # Convert to meters
                    self.get_logger().info(f'3D point at pixel ({x}, {y}): X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f} meters')
                else:
                    self.get_logger().warn(f'Point contains NaN values or insufficient data: {point}')
                    return None
            else:
                self.get_logger().warn(f'Point index {point_index} is out of range for {len(points_list)} points')
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
        
        self.get_logger().info(f"Plane normal: {normal}")
        self.get_logger().info(f"Plane center: {center}")
        self.get_logger().info(f"Is planar: {is_planar}")
        
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
        if "robot_center" not in plane_info or "robot_normal" not in plane_info:
            self.get_logger().error("Plane information not available in robot frame")
            return None
            
        # Get position (already in robot frame)
        position = plane_info["robot_center"]
        
        # Calculate orientation based on normal vector
        orientation = self.calculate_tcp_orientation_from_normal(plane_info["robot_normal"])
        
        tcp_pose = {
            "position": position,
            "orientation": orientation
        }
        
        # Log the information
        self.get_logger().info(f"TCP Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        self.get_logger().info(f"TCP Orientation (quaternion): [{orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}, {orientation[3]:.4f}]")
        
        return tcp_pose

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()