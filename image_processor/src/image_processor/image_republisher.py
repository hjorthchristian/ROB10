import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class BgrToRgbConverter(Node):
    def __init__(self):
        super().__init__('image_republisher')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Original BGR topic
            self.callback,
            80)
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_rgb',  # New RGB topic
            30)

    def callback(self, msg):
        try:
            # Parse the incoming image message
            height = msg.height
            width = msg.width
            encoding = msg.encoding
            step = msg.step
            
            # Convert byte array to numpy array
            array = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
            
            # Switch BGR to RGB by swapping the channels
            rgb_array = array[:, :, ::-1].copy()  # Simple RGB conversion by reversing the order
            
            # Create new Image message
            rgb_msg = Image()
            rgb_msg.header = msg.header
            rgb_msg.height = height
            rgb_msg.width = width
            rgb_msg.encoding = 'rgb8'
            rgb_msg.step = width * 3
            rgb_msg.data = rgb_array.tobytes()
            
            # Publish RGB image
            self.publisher.publish(rgb_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    converter = BgrToRgbConverter()
    rclpy.spin(converter)
    converter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()