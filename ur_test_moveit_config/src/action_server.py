#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from control_msgs.action import GripperCommand
from ur_msgs.srv import SetIO
import time

class VacuumGripperController(Node):
    def __init__(self):
        super().__init__('ur_vacuum_controller')
        
        # Use a callback group to allow concurrent callbacks
        callback_group = ReentrantCallbackGroup()
        
        self._action_server = ActionServer(
            self,
            GripperCommand,
            '/ur_vacuum_controller/gripper_cmd',
            self.execute_callback,
            callback_group=callback_group)
            
        self.set_io_client = self.create_client(
            SetIO, 
            '/io_and_status_controller/set_io',
            callback_group=callback_group)
            
        # Wait for the service to become available
        while not self.set_io_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for SetIO service...')
            
        self.get_logger().info('Vacuum gripper controller has been started')
        
    async def execute_callback(self, goal_handle):
        self.get_logger().info('Received gripper command')
        request = goal_handle.request
        
        # Typically for vacuum grippers:
        # position > 0.5 means grip (vacuum on)
        # position <= 0.5 means release (vacuum off)
        if request.command.position > 0.5:
            self.get_logger().info('Turning vacuum ON')
            await self.call_set_io(1, 0, 1)  # Digital output, pin 0, ON
        else:
            self.get_logger().info('Turning vacuum OFF')
            await self.call_set_io(1, 0, 0)  # Digital output, pin 0, OFF
            
        # Short delay to allow the vacuum to change state
        time.sleep(0.5)
            
        goal_handle.succeed()
        result = GripperCommand.Result()
        return result
        
    async def call_set_io(self, fun, pin, state):
        request = SetIO.Request()
        # Use integers for the UR driver
        request.fun = int(fun)     # 1 = digital output
        request.pin = int(pin)     # Pin number (0-based)
        request.state = float(state) # 0 = OFF, 1 = ON
        
        self.get_logger().info(f'Setting IO: fun={fun}, pin={pin}, state={state}')
        
        # Use await to properly wait for the service response
        future = self.set_io_client.call_async(request)
        await future
        
        if future.result() is not None:
            self.get_logger().info(f'SetIO response: {future.result().success}')
            return future.result().success
        else:
            self.get_logger().error('Service call failed')
            return False

def main():
    rclpy.init()
    node = VacuumGripperController()
    
    # Use a MultiThreadedExecutor for better performance with async calls
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        node.get_logger().info('Starting vacuum gripper controller')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()