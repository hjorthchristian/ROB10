#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
from sensor_msgs.msg import Image
import numpy as np
import time
import cv2
import os
from datetime import datetime

class DepthConsistencyMonitor(Node):
    def __init__(self):
        super().__init__('depth_consistency_monitor')
        
        # Create callback groups for concurrent execution
        self.subscription_group = MutuallyExclusiveCallbackGroup()
        
        # Image storage with threading lock
        self.image_lock = threading.Lock()
        self.rgb_image = None
        self.depth_image = None
        
        # Target pixels to monitor (x, y) coordinates
        self.target_pixels = [
            (518, 38), 
            (561, 76), 
            (489, 62), 
            (541, 107)
        ]
        
        # Data storage for each pixel
        self.pixel_data = {}
        for pixel in self.target_pixels:
            self.pixel_data[pixel] = {
                'values': [],
                'timestamps': [],
                'mean': 0.0,
                'std_dev': 0.0,
                'min': float('inf'),
                'max': 0.0,
                'count': 0,            # Total count of measurements
                'avg_5': 0.0,          # Average of last 5 values
                'avg_10': 0.0,         # Average of last 10 values
                'avg_20': 0.0          # Average of last 20 values
            }
        
        # Output file for depth values
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.expanduser("~/depth_data")
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = f"{self.output_dir}/depth_values_{timestamp}.txt"
        
        with open(self.output_file, 'w') as f:
            header = "timestamp,"
            for pixel in self.target_pixels:
                header += f"depth_{pixel[0]}_{pixel[1]},count_{pixel[0]}_{pixel[1]},avg5_{pixel[0]}_{pixel[1]},avg10_{pixel[0]}_{pixel[1]},avg20_{pixel[0]}_{pixel[1]},"
            header += "non_black_mean,non_black_std,non_black_min,non_black_max,non_black_count\n"
            f.write(header)
        
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
        
        # Timer for periodic statistics display
        self.stats_timer = self.create_timer(1.0, self.display_statistics)
        
        # Depth data collection timer
        self.collection_timer = self.create_timer(0.1, self.process_depth_data)
        
        # Non-black pixels depth data
        self.non_black_depths = []
        self.last_save_time = time.time()
        self.save_interval = 5.0  # Save to file every 5 seconds
        
        self.get_logger().info('=========================================')
        self.get_logger().info('Depth Consistency Monitor initialized')
        self.get_logger().info(f'Monitoring pixels: {self.target_pixels}')
        self.get_logger().info(f'Saving data to: {self.output_file}')
        self.get_logger().info('=========================================')
    
    def calculate_moving_average(self, values, window_size):
        """Calculate moving average for the last 'window_size' values."""
        # Check if the array is empty or None using proper NumPy methods
        if values is None or len(values) == 0:
            return 0.0
        
        # values is already a NumPy array in your case, but let's ensure it
        values_array = values if isinstance(values, np.ndarray) else np.array(values)
        
        if len(values_array) < window_size:
            return np.mean(values_array)
        return np.mean(values_array[-window_size:])
    
    def rgb_callback(self, msg):
        """RGB image callback."""
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
            except Exception as e:
                self.get_logger().error(f'Error processing RGB image: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
    
    def depth_callback(self, msg):
        """Depth image callback."""
        with self.image_lock:
            try:
                if msg.encoding != '16UC1':
                    self.get_logger().error(f'Unsupported depth encoding: {msg.encoding}')
                    return

                depth_data = np.frombuffer(msg.data, dtype=np.uint16)
                self.depth_image = depth_data.reshape((msg.height, msg.width))
            except Exception as e:
                self.get_logger().error(f'Error processing depth image: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
    
    def process_depth_data(self):
        """Process depth data for target pixels and non-black regions."""
        if self.rgb_image is None or self.depth_image is None:
            return
            
        with self.image_lock:
            try:
                # Create a local copy to avoid race conditions
                rgb_copy = self.rgb_image.copy()
                depth_copy = self.depth_image.copy()
                
                current_time = time.time()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Process target pixels
                line_data = f"{timestamp},"
                for pixel in self.target_pixels:
                    x, y = pixel
                    
                    # Check if the pixel is within bounds
                    if (x < depth_copy.shape[1] and y < depth_copy.shape[0] and 
                        x >= 0 and y >= 0):
                        depth_value = depth_copy[y, x]
                        
                        # Increment the count only for valid depth values
                        if depth_value > 0 and depth_value < 10000:
                            self.pixel_data[pixel]['count'] += 1
                        
                        # Store the data
                        self.pixel_data[pixel]['values'].append(depth_value)
                        self.pixel_data[pixel]['timestamps'].append(current_time)
                        
                        # Keep only last 100 values
                        if len(self.pixel_data[pixel]['values']) > 100:
                            self.pixel_data[pixel]['values'].pop(0)
                            self.pixel_data[pixel]['timestamps'].pop(0)
                        
                        # Update statistics
                        values = np.array(self.pixel_data[pixel]['values'])
                        if len(values) > 0:
                            self.pixel_data[pixel]['mean'] = np.mean(values)
                            self.pixel_data[pixel]['std_dev'] = np.std(values)
                            self.pixel_data[pixel]['min'] = np.min(values)
                            self.pixel_data[pixel]['max'] = np.max(values)
                            
                            # Calculate moving averages
                            self.pixel_data[pixel]['avg_5'] = self.calculate_moving_average(values, 5)
                            self.pixel_data[pixel]['avg_10'] = self.calculate_moving_average(values, 10)
                            self.pixel_data[pixel]['avg_20'] = self.calculate_moving_average(values, 20)
                        
                        # Add data to the line
                        count = self.pixel_data[pixel]['count']
                        avg5 = self.pixel_data[pixel]['avg_5']
                        avg10 = self.pixel_data[pixel]['avg_10']
                        avg20 = self.pixel_data[pixel]['avg_20']
                        line_data += f"{depth_value},{count},{avg5:.1f},{avg10:.1f},{avg20:.1f},"
                    else:
                        self.get_logger().warn(f"Pixel {pixel} is out of bounds")
                        line_data += "out_of_bounds,0,0,0,0,"
                
                # Process non-black pixels
                # Create a mask where RGB pixels are not black
                non_black_mask = np.any(rgb_copy > 0, axis=2)
                
                # Get depth values for non-black pixels
                non_black_depths = depth_copy[non_black_mask]
                
                # Filter out zero or invalid depths
                valid_depths = non_black_depths[(non_black_depths > 0) & (non_black_depths < 10000)]
                
                if len(valid_depths) > 0:
                    mean_depth = np.mean(valid_depths)
                    std_dev = np.std(valid_depths)
                    min_depth = np.min(valid_depths)
                    max_depth = np.max(valid_depths)
                    
                    # Add to the line data
                    line_data += f"{mean_depth},{std_dev},{min_depth},{max_depth},{len(valid_depths)}\n"
                    
                    # Store for later analysis
                    self.non_black_depths = valid_depths
                else:
                    line_data += "no_valid_depths,0,0,0,0\n"
                
                # Write to file
                with open(self.output_file, 'a') as f:
                    f.write(line_data)
                
                # Save full depth data periodically
                if current_time - self.last_save_time > self.save_interval:
                    self.save_full_depth_data(depth_copy, non_black_mask)
                    self.last_save_time = current_time
                
            except Exception as e:
                self.get_logger().error(f'Error processing depth data: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
    
    def save_full_depth_data(self, depth_image, non_black_mask):
        """Save full depth data for non-black pixels to a separate file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.output_dir}/full_depth_{timestamp}.txt"
            
            # Get coordinates and depths of non-black pixels
            y_coords, x_coords = np.where(non_black_mask)
            depths = depth_image[non_black_mask]
            
            with open(output_file, 'w') as f:
                f.write("x,y,depth\n")
                for x, y, d in zip(x_coords, y_coords, depths):
                    if d > 0 and d < 10000:  # Filter valid depths
                        f.write(f"{x},{y},{d}\n")
            
            self.get_logger().info(f"Saved full depth data to {output_file}")
        except Exception as e:
            self.get_logger().error(f'Error saving full depth data: {e}')
    
    def display_statistics(self):
        """Display current statistics for monitored pixels."""
        if not any(len(data['values']) > 0 for data in self.pixel_data.values()):
            self.get_logger().info("No depth data collected yet")
            return
            
        self.get_logger().info("=== DEPTH PIXEL STATISTICS ===")
        
        for pixel, data in self.pixel_data.items():
            if len(data['values']) > 0:
                self.get_logger().info(
                    f"Pixel ({pixel[0]}, {pixel[1]}): "
                    f"Current={data['values'][-1]}mm, "
                    f"Count={data['count']}, "
                    f"Mean={data['mean']:.1f}mm, "
                    f"StdDev={data['std_dev']:.1f}mm, "
                    f"Min={data['min']}mm, "
                    f"Max={data['max']}mm, "
                    f"Range={data['max'] - data['min']}mm"
                )
                self.get_logger().info(
                    f"  Moving Avgs: "
                    f"Last 5={data['avg_5']:.1f}mm, "
                    f"Last 10={data['avg_10']:.1f}mm, "
                    f"Last 20={data['avg_20']:.1f}mm"
                )
        
        if len(self.non_black_depths) > 0:
            self.get_logger().info(
                f"Non-black pixels: "
                f"Mean={np.mean(self.non_black_depths):.1f}mm, "
                f"StdDev={np.std(self.non_black_depths):.1f}mm, "
                f"Min={np.min(self.non_black_depths)}mm, "
                f"Max={np.max(self.non_black_depths)}mm, "
                f"Count={len(self.non_black_depths)}"
            )
        
        self.get_logger().info("==============================")

def main(args=None):
    rclpy.init(args=args)
    
    # Create the node
    depth_monitor = DepthConsistencyMonitor()
    
    # Use a multithreaded executor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(depth_monitor)
    
    try:
        depth_monitor.get_logger().info('Starting depth consistency monitoring...')
        executor.spin()
    except KeyboardInterrupt:
        depth_monitor.get_logger().info("Monitor interrupted by user")
    except Exception as e:
        depth_monitor.get_logger().error(f"Error during monitoring: {e}")
        import traceback
        depth_monitor.get_logger().error(traceback.format_exc())
    finally:
        # Clean up
        depth_monitor.get_logger().info("Shutting down monitor...")
        depth_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()