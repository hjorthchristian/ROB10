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
        (2, 1, 2),  # length, width, height
        (2, 1, 2),
        (2, 2, 1),
        (3, 1, 1),
        (1, 3, 1)
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
                print(f"Box placed at position: ({pos.x}, {pos.y}, {pos.z})")
                
                # Display the box dimensions along x and y axes
                print(f"Box dimensions on pallet: {response.x_dimension}x{response.y_dimension}")
                
                if response.orientations:
                    orient = response.orientations[0]
                    print(f"Box orientation quaternion: ({orient.x}, {orient.y}, {orient.z}, {orient.w})")
                    # Indicate if box is rotated based on quaternion z component
                    is_rotated = orient.z > 0
                    print(f"Box is {'rotated' if is_rotated else 'not rotated'} on the pallet")
            else:
                print("Failed to place box")
        else:
            client.get_logger().error(f'Service call failed: {future.exception()}')
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()