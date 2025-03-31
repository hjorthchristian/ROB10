#!/home/chrishj/ros2_env/bin/python3
import sys
import os
print(f"Python interpreter: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"Environment PATH: {os.environ.get('PATH')}")
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import cv2
import numpy as np
import struct
import sensor_msgs_py.point_cloud2 as pc2
import math
from ultralytics import YOLOE

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
        self.model = YOLOE("yoloe-11s-seg.pt")
        self.objects = ["package", "box"]
        self.model.set_classes(self.objects, self.model.get_text_pe(self.objects))  
        self.cv_image = None
        self.point_cloud_data = None
        
        # Image resolutions
        self.rgb_width, self.rgb_height = 1280, 720
        self.depth_width, self.depth_height = 848, 480
        
        # Target pixel in RGB image
        self.target_rgb_x, self.target_rgb_y = 762, 290
        
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
                        self.get_logger().info(f"Class names: {result.names}")
                    
                    # Check if boxes are available
                    if hasattr(result, 'boxes'):
                        boxes_data = result.boxes
                        num_detections = len(boxes_data)
                        self.get_logger().info(f"Number of detections: {num_detections}")
                        
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
                                    self.target_depth_x = depth_x
                                    self.target_depth_y = depth_y
                                
                                self.get_logger().info(
                                    f"Detection {i}: {class_name} (conf: {conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]"
                                )
                    
                    # Check for segmentation masks and visualize if available
                    if hasattr(result, 'masks') and result.masks is not None:
                        self.get_logger().info(f"Masks attributes: {dir(result.masks)}")
                        
                        if hasattr(result.masks, 'data') and result.masks.data is not None:
                            masks = result.masks.data
                            self.get_logger().info(f"Masks shape: {masks.shape}")
                            
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
                self.get_point_from_cloud(self.target_depth_x, self.target_depth_y)
                
        except Exception as e:
            self.get_logger().error(f'PointCloud processing error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

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
                else:
                    self.get_logger().warn(f'Point contains NaN values or insufficient data: {point}')
            else:
                self.get_logger().warn(f'Point index {point_index} is out of range for {len(points_list)} points')
                    
        except Exception as e:
            self.get_logger().error(f'Error extracting point: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()