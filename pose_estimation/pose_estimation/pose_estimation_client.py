import rclpy
from rclpy.node import Node
import sys
import argparse
from pose_estimation_interfaces.srv import PoseEstimation
import numpy as np


class PoseEstimationClient(Node):
    def __init__(self, text_prompt):
        super().__init__('pose_estimation_client')
        
        # Create client
        self.client = self.create_client(
            PoseEstimation, 
            'estimate_pose'
        )
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        
        # Store parameters
        self.text_prompt = text_prompt
        
        # Send request
        self.send_request()
    
    def send_request(self):
        """Send service request with text prompt."""
        self.get_logger().info('Preparing request...')
        
        try:
            # Create request
            request = PoseEstimation.Request()
            request.text_prompt = self.text_prompt
            
            # Send request
            self.get_logger().info(f'Sending request with prompt: "{self.text_prompt}"')
            future = self.client.call_async(request)
            
            # Add callback for when response is received
            future.add_done_callback(self.response_callback)
            
        except Exception as e:
            self.get_logger().error(f'Error preparing request: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def response_callback(self, future):
        """Process the service response."""
        try:
            response = future.result()
            
            if response.success:
                self.get_logger().info('Pose estimation successful!')
                
                # Print pose information
                self.get_logger().info('Grasp pose information:')
                self.get_logger().info(f'Position: [{response.position.x:.4f}, {response.position.y:.4f}, {response.position.z:.4f}]')
                for i, orientation in enumerate(response.orientations):
                    yaw_angle = i * 90  # 0, 90, 180, 270 degrees
                    self.get_logger().info(f'Orientation {i+1} (yaw={yaw_angle}°): '
                                        f'[{orientation.x:.4f}, {orientation.y:.4f}, '
                                        f'{orientation.z:.4f}, {orientation.w:.4f}]')
                self.get_logger().info(f'x_width: {response.x_width:.4f}, y_length: {response.y_length:.4f}, z_height: {response.z_height:.4f}')
                
            else:
                self.get_logger().error(f'Service failed: {response.error_message}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing response: {e}')
        finally:
            # Shutdown ROS
            self.get_logger().info('Client finished, shutting down...')
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Client for pose estimation service')
    parser.add_argument('--prompt', type=str, default='box in the top middle', help='Text prompt for segmentation')
    
    args, unknown = parser.parse_known_args()
    
    # Initialize ROS
    rclpy.init(args=unknown)
    
    # Create and run client
    client = PoseEstimationClient(
        text_prompt=args.prompt
    )
    
    # Spin until response callback completes
    rclpy.spin(client)


if __name__ == '__main__':
    main()