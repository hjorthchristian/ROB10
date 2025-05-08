import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32, PolygonStamped, Polygon, Point
from std_msgs.msg import Header
import numpy as np
import cv2
from PIL import Image as PILImage

from lang_sam import LangSAM
from lang_sam_interfaces.srv import SegmentImage  # Your custom service message

import os
import datetime

class LangSAMServer(Node):
    def __init__(self):
        super().__init__('lang_sam_server')
        self.srv = self.create_service(
            SegmentImage, 
            'segment_image', 
            self.segment_callback)
        self.model = LangSAM(sam_type="sam2.1_hiera_large")
        self.get_logger().info('LangSAM Service ready')
        self.declare_parameter('save_images', True)
        #self.declare_parameter('save_directory', '/lang_sam_segmentations')
        self.save_images = self.get_parameter('save_images').value
        #self.save_directory = self.get_parameter('save_directory').value

        
    def segment_callback(self, request, response):
        # Convert ROS Image to PIL Image without using CvBridge
        if request.image.encoding != 'rgb8':
            self.get_logger().error(f'Unsupported encoding: {request.image.encoding}')
            return response
            
        # Create a numpy array from the image data
        # For RGB8 format, each pixel is 3 bytes
        image_data = np.frombuffer(request.image.data, dtype=np.uint8)
        
        # Reshape the array to the image dimensions
        # ROS uses row-major ordering
        cv_image = image_data.reshape((request.image.height, request.image.width, 3))
        
        # Convert to PIL Image format
        pil_image = PILImage.fromarray(cv_image)
        
        # Run segmentation with text prompt
        results = self.model.predict([pil_image], [request.text_prompt])
        result = results[0]  # First image result
        
        # Prepare response header
        response.header = request.image.header
        
        # Original image with overlaid masks
        overlay_image = cv_image.copy()
        
        # Process each detected object
        for i in range(len(result['labels'])):
            mask = result['masks'][i]
            label = result['labels'][i]
            score = float(result['scores'][i])

            if score < 0.5:
                continue
            
            # Add to response arrays
            response.labels.append(label)
            response.scores.append(score)
            
            
            # Convert mask to ROS Image without CvBridge
            mask_binary = (mask * 255).astype(np.uint8)
            mask_msg = Image()
            mask_msg.header = request.image.header
            mask_msg.height = mask_binary.shape[0]
            mask_msg.width = mask_binary.shape[1]
            mask_msg.encoding = 'mono8'
            mask_msg.is_bigendian = False
            mask_msg.step = mask_binary.shape[1]  # Full row length in bytes
            mask_msg.data = mask_binary.tobytes()  # Convert to bytes
            
            response.mask_images.append(mask_msg)
            
            # Find contour of mask
            contours, _ = cv2.findContours(
                mask_binary, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Create PolygonStamped message
                poly_msg = PolygonStamped()
                poly_msg.header = request.image.header
                
                largest_contour = max(contours, key=cv2.contourArea)
                # Simplify contour to reduce message size
                approx = cv2.approxPolyDP(
                    largest_contour, 
                    0.01 * cv2.arcLength(largest_contour, True), 
                    True
                )
                
                for point in approx.reshape(-1, 2):
                    p = Point32()
                    p.x = float(point[0])
                    p.y = float(point[1])
                    p.z = 0.0
                    poly_msg.polygon.points.append(p)
                
                response.contours.append(poly_msg)
                
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    centroid = Point()
                    centroid.x = float(cx)
                    centroid.y = float(cy)
                    centroid.z = 0.0  # Could be filled with depth data if available
                    response.centroids.append(centroid)
                    
                    # Draw on overlay image
                    color = np.random.randint(0, 255, size=3).tolist()
                    cv2.drawContours(overlay_image, [largest_contour], -1, color, 2)
                    cv2.putText(
                        overlay_image, 
                        f"{label}: {score:.2f}", 
                        (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        1
                    )
        
        # Convert overlay image to ROS Image without CvBridge
        segmented_msg = Image()
        segmented_msg.header = request.image.header
        segmented_msg.height = overlay_image.shape[0]
        segmented_msg.width = overlay_image.shape[1]
        segmented_msg.encoding = 'rgb8'
        segmented_msg.is_bigendian = False
        segmented_msg.step = overlay_image.shape[1] * 3  # Full row length in bytes (3 bytes per pixel for RGB8)
        segmented_msg.data = overlay_image.tobytes()  # Convert to bytes
        
        response.segmented_image = segmented_msg
        
        self.get_logger().info(f'Processed segmentation request, found {len(response.labels)} objects')
        
        if self.save_images and response.labels:
            # Create a timestamp for the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a string with object names (limit to first 3 objects if there are many)
            if len(response.labels) <= 3:
                objects_str = "_".join(response.labels)
            else:
                objects_str = "_".join(response.labels[:3]) + "_and_more"
            
            # Limit the length of objects_str to avoid excessively long filenames
            if len(objects_str) > 100:
                objects_str = objects_str[:100]
            
            # Create sanitized filename (replace invalid characters)
            objects_str = objects_str.replace('/', '_').replace(' ', '_')
            
            # Create full filename
            filename = f"{timestamp}_{objects_str}.png"
            
            # Save the image - convert RGB to BGR for OpenCV
            cv2.imwrite(filename, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f'Saved segmented image to {filename}')
        return response

def main(args=None):
    rclpy.init(args=args)
    lang_sam_server = LangSAMServer()
    rclpy.spin(lang_sam_server)
    rclpy.shutdown()

if __name__ == '__main__':
    main()