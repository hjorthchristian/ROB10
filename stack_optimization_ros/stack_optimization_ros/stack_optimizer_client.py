#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from stack_optimization_interfaces.srv import StackOptimizer
from geometry_msgs.msg import Quaternion


class BoxStackClient(Node):
    def __init__(self):
        super().__init__('box_stack_client')
        self.client = self.create_client(StackOptimizer, 'box_stack_optimizer')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Box stack service not available, waiting...')
        
        self.get_logger().info('Box stack service is now available')
    
    def send_request(self, length, width, height, change_stack_allowed=False):
        """Send a request to place a box"""
        request = StackOptimizer.Request()
        request.length = length
        request.width = width
        request.height = height
        request.orientation = Quaternion()  # Default orientation (identity)
        request.change_stack_allowed = change_stack_allowed
        
        future = self.client.call_async(request)
        return future


def main(args=None):
    rclpy.init(args=args)
    client = BoxStackClient()
    
    # Example boxes to place
    boxes = [
        (40, 20, 20),
        (20, 20, 20),
      
    ]
    
    for i, (l, w, h) in enumerate(boxes):
        # For the first 3 boxes, allow changing the stack
        # For the remaining boxes, don't allow changes
        change_allowed = False
        
        print(f"Placing box {i+1}: {l}x{w}x{h} (change_allowed={change_allowed})")
        future = client.send_request(l, w, h, change_allowed)
        
        rclpy.spin_until_future_complete(client, future)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                pos = response.position
                print(f"Box placed at global position (meters): ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")
                
                # Display the box dimensions
                print(f"Box dimensions (optimizer units): {response.x_dimension}x{response.y_dimension}")
                
                # We don't have direct access to the scaling factors here, but we can inform the user
                print(f"NOTE: The box dimensions above are in optimizer units.")
                print(f"      To convert to meters, these would need to be scaled by the correct factors.")
                
                if response.orientations:
                    orient = response.orientations[0]
                    print(f"Box orientation quaternion: ({orient.x:.4f}, {orient.y:.4f}, {orient.z:.4f}, {orient.w:.4f})")
                    # Indicate if box is rotated based on quaternion z component
                    is_rotated = orient.z > 0
                    print(f"Box is {'rotated' if is_rotated else 'not rotated'} on the pallet")
                    
                print(f"\nCoordinate System Information:")
                print(f"- Global coordinates are in METERS")
                print(f"- The pallet's top-left corner is at global position (0.5792, 0.4032, -0.8384)")
                print(f"- Box positions have been transformed from optimizer units to the global coordinate system")
                print(f"- Proper scaling has been applied to convert between unit systems")
            else:
                print("Failed to place box")
        else:
            client.get_logger().error(f'Service call failed: {future.exception()}')
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()