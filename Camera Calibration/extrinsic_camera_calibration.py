#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
import time
import os
import cv2
import threading
import sys

class TerminalDataCollector(Node):
    def __init__(self):
        super().__init__('terminal_data_collector')
        
        # Camera subscriber
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Update this with your actual camera topic
            self.camera_callback,
            10
        )
        
        # TF2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Status variables
        self.frame_count = 0
        self.latest_image = None
        self.running = True
        
        # Create output directory
        self.output_dir = "calibration_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.get_logger().info("Terminal-Based Data Collector started")
        self.get_logger().info("Type 's' and press Enter to save a frame")
        self.get_logger().info("Type 'q' and press Enter to quit")
        self.get_logger().info("Type 't' and press Enter to print current transform")
        
        # Create a timer for showing transforms periodically
        self.transform_timer = self.create_timer(30.0, self.print_transform)
        
        # Start input thread
        self.input_thread = threading.Thread(target=self.input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
        
    def input_loop(self):
        """Thread function to handle user input"""
        while self.running:
            try:
                # Get input from user
                cmd = input("Enter command (s=save, t=transform, q=quit): ").strip().lower()
                
                if cmd == 's':
                    # Save frame
                    self.get_logger().info("Save command received")
                    self.save_data_frame()
                elif cmd == 't':
                    # Print transform
                    self.get_logger().info("Transform command received")
                    self.print_transform()
                elif cmd == 'q':
                    # Quit
                    self.get_logger().info(f"Quit command received. Collected {self.frame_count} frames.")
                    self.running = False
                    rclpy.shutdown()
                    break
                else:
                    self.get_logger().info(f"Unknown command: {cmd}")
            except Exception as e:
                self.get_logger().error(f"Input error: {e}")
                time.sleep(1)  # Sleep briefly to avoid high CPU usage
    
    def print_transform(self):
        """Print the current transform"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'ur10_base_link',
                'ur10_wrist_3_link',
                rclpy.time.Time()
            )
            # Print transform
            self.get_logger().info("Current transform:")
            self.get_logger().info(f"Translation: [{transform.transform.translation.x}, {transform.transform.translation.y}, {transform.transform.translation.z}]")
            self.get_logger().info(f"Rotation: [{transform.transform.rotation.x}, {transform.transform.rotation.y}, {transform.transform.rotation.z}, {transform.transform.rotation.w}]")
            return transform
        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")
            return None
        
    def camera_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def save_data_frame(self):
        """Save a frame and its corresponding transform"""
        if self.latest_image is None:
            self.get_logger().warn("No image available")
            return False
        
        try:
            # Get current robot pose
            transform = self.print_transform()
            if transform is None:
                return False
                
            # Extract transform data
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            
            # Save image
            self.frame_count += 1
            img_filename = os.path.join(self.output_dir, f"frame_{self.frame_count:03d}.png")
            cv2.imwrite(img_filename, self.latest_image)
            
            # Save transform data
            transform_filename = os.path.join(self.output_dir, f"transform_{self.frame_count:03d}.txt")
            with open(transform_filename, 'w') as f:
                f.write(f"Translation: {translation.x} {translation.y} {translation.z}\n")
                f.write(f"Rotation: {rotation.x} {rotation.y} {rotation.z} {rotation.w}\n")
            
            # Print success message
            self.get_logger().info(f"Frame {self.frame_count} captured and saved to {img_filename}!")
            self.get_logger().info(f"Transform saved to {transform_filename}")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error capturing frame: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    data_collector = TerminalDataCollector()
    
    try:
        rclpy.spin(data_collector)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        data_collector.running = False
        data_collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()