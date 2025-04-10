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
#from PIL import Image


class LangSAMClient(Node):
    def __init__(self, text_prompt="box", save_path=None, display_time=30):
        super().__init__('lang_sam_client')
        
        # Initialize service client
        self.client = self.create_client(SegmentImage, 'segment_image')
        
        # Store parameters
        self.text_prompt = text_prompt
        self.save_path = save_path
        self.display_time = display_time  # How long to display the image (seconds)
        
        # Flag to track if we've processed an image
        self.image_processed = False
        
        # Create subscription to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        
        self.latest_image = None
        
        # Configure matplotlib to use a non-interactive backend that doesn't require GTK
        plt.switch_backend('agg')  # Use the non-interactive Agg backend
        
        self.get_logger().info(f'LangSAM Client initialized with prompt: "{text_prompt}"')
        self.get_logger().info('Waiting for one image on /camera/camera/color/image_raw...')
    
    def image_callback(self, msg):
        # Skip if we've already processed an image
        if self.image_processed:
            return
            
        self.get_logger().info('Received image, processing...')
        
        try:
            # Convert ROS Image to NumPy array without using CvBridge
            if msg.encoding != 'rgb8':
                self.get_logger().warn(f'Expected rgb8 encoding, got {msg.encoding}. Attempting to process anyway.')
                
            # Create numpy array from image data
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            # Reshape the array to image dimensions
            cv_image = image_data.reshape((msg.height, msg.width, 3))
            
            # Store the latest image for visualization
            self.latest_image = cv_image
            
            # Mark that we've processed an image to prevent further processing
            self.image_processed = True
            
            # Send segmentation request
            self.send_request(msg, self.text_prompt)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            # Shutdown if there's an error
            rclpy.shutdown()
            
    def send_request(self, ros_img, text_prompt):
        # Create request
        request = SegmentImage.Request()
        request.image = ros_img
        request.text_prompt = text_prompt
        
        # Wait for service
        if not self.client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Service not available after timeout')
            rclpy.shutdown()
            return
        
        # Send request
        future = self.client.call_async(request)
        future.add_done_callback(self.process_response_callback)
        
    def process_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Found {len(response.labels)} objects')
            
            if self.latest_image is None:
                self.get_logger().error('No image available for visualization')
                rclpy.shutdown()
                return
                
            # Create a copy of the original image for visualization
            visualized_image = self.latest_image.copy()
            
            # Access all detected objects with their data
            for i in range(len(response.labels)):
                label = response.labels[i]
                score = response.scores[i]
                centroid = response.centroids[i]
                
                self.get_logger().info(
                    f'Object {i}: {label} (score: {score:.2f}) at position '
                    f'({centroid.x}, {centroid.y})'
                )
                
                # Convert mask from ROS Image to NumPy array without CvBridge
                mask_data = np.frombuffer(response.mask_images[i].data, dtype=np.uint8)
                mask = mask_data.reshape((response.mask_images[i].height, response.mask_images[i].width))
                
                # Generate random color for this instance
                color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                
                # Create colored mask overlay
                colored_mask = np.zeros_like(visualized_image)
                for c in range(3):
                    colored_mask[:, :, c] = np.where(mask > 0, color[c], 0).astype(np.uint8)
                
                # Blend mask with image
                alpha = 0.5  # Transparency factor
                mask_bool = mask > 0
                for c in range(3):
                    visualized_image[:, :, c] = np.where(
                        mask_bool,
                        visualized_image[:, :, c] * (1 - alpha) + colored_mask[:, :, c] * alpha,
                        visualized_image[:, :, c]
                    )
            
            # Save image
            save_path = self.save_path or 'lang_sam_result.png'
            
            plt.figure(figsize=(12, 8))
            plt.imshow(visualized_image)
            plt.axis('off')
            plt.title(f"Detected {self.text_prompt}")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            self.get_logger().info(f'Saved result to {save_path}')
            cv_visualized_image = cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB)
            self.get_logger().info(f'Image shape: {cv_visualized_image.shape}')
            cv2.imshow('Result', cv_visualized_image)
            cv2.waitKey(15500)  # Display for 2.5 seconds
            cv2.destroyAllWindows()
            #pil_image = Image.fromarray(visualized_image)
            #pil_image.show() 
            
            # Display message to user about viewing the image
            self.get_logger().info(f'Image saved to {save_path}')
            self.get_logger().info(f'To view the image, open it with an image viewer:')
            self.get_logger().info(f'  - For example: xdg-open {save_path}')
            self.get_logger().info(f'  - Or: display {save_path}')
            self.get_logger().info(f'  - Or: eog {save_path}')
            
            # Wait a moment before shutting down
            self.get_logger().info(f'Shutting down in {self.display_time} seconds...')
            
            # Create a timer to delay shutdown
            threading.Timer(self.display_time, self.shutdown_node).start()
                
        except Exception as e:
            self.get_logger().error(f'Error processing results: {e}')
            rclpy.shutdown()
    
    def shutdown_node(self):
        # Shutdown ROS2 after displaying/saving the image
        self.get_logger().info('Processing complete. Shutting down...')
        rclpy.shutdown()

def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LangSAM Client for ROS2 (one-shot mode)')
    parser.add_argument('--prompt', type=str, default='box on the right', help='Text prompt for segmentation')
    parser.add_argument('--save', type=str, help='Path to save the result image')
    parser.add_argument('--wait', type=int, default=10, help='Time to wait before shutting down (seconds)')
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Parse args
    if args is None:
        args = sys.argv[1:]
    parsed_args = parser.parse_args(args)
    
    # Create client node
    client_node = LangSAMClient(
        text_prompt=parsed_args.prompt,
        save_path=parsed_args.save,
        display_time=parsed_args.wait
    )
    
    try:
        # Spin the node to process callbacks
        rclpy.spin(client_node)
    
    except KeyboardInterrupt:
        client_node.get_logger().info('Interrupted by user')
    
    finally:
        # Cleanup
        client_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()