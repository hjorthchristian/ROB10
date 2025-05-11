import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import asyncio
import time
import signal
import sys
from ollama_ros_interfaces.action import ProcessThinking

class Qwen3Client(Node):
    def __init__(self):
        super().__init__('qwen3_client')
        self._client = ActionClient(self, ProcessThinking, 'process_thinking')
        
        # Add buffer and timing variables for throttling
        self._thinking_buffer = ""
        self._response_buffer = ""
        self._last_log_time = 0
        self._log_interval = 0.5  # Log every 0.5 seconds
        
    async def send_prompt(self, model_name, prompt):
        # Wait for server
        self.get_logger().info('Waiting for action server...')
        self._client.wait_for_server()
        
        # Create goal
        goal_msg = ProcessThinking.Goal()
        goal_msg.model_name = model_name
        goal_msg.prompt = prompt
        
        self.get_logger().info(f'Sending prompt to {model_name}...')
        
        # Send goal with feedback callback
        self._send_goal_future = self._client.send_goal_async(
            goal_msg, 
            feedback_callback=self._feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self._goal_response_callback)
        
    def _feedback_callback(self, feedback_msg):
        current_time = time.time()
        
        # Add new content to appropriate buffer
        if feedback_msg.feedback.is_thinking:
            self._thinking_buffer += feedback_msg.feedback.partial_content
        else:
            self._response_buffer += feedback_msg.feedback.partial_content
        
        # Only log if enough time has passed since last log
        if current_time - self._last_log_time >= self._log_interval:
            # Log thinking content if any
            if self._thinking_buffer:
                self.get_logger().info(f"Thinking: {self._thinking_buffer}")
                self._thinking_buffer = ""  # Clear buffer after logging
                
            # Log response content if any
            if self._response_buffer:
                self.get_logger().info(f"Responding: {self._response_buffer}")
                self._response_buffer = ""  # Clear buffer after logging
                
            self._last_log_time = current_time
    
    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
            
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_callback)
        
    def _get_result_callback(self, future):
        # Log any remaining buffered content before showing the final result
        if self._thinking_buffer:
            self.get_logger().info(f"Thinking: {self._thinking_buffer}")
            self._thinking_buffer = ""
            
        if self._response_buffer:
            self.get_logger().info(f"Responding: {self._response_buffer}")
            self._response_buffer = ""
            
        result = future.result().result
        self.get_logger().info(f'Final answer: {result.answer}')

def main(args=None):
    # Setup signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print('\nCaught Ctrl+C! Shutting down...')
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    rclpy.init(args=args)
    client = Qwen3Client()
    
    try:
        # Define example prompt
        model_name = "qwen3:1.7b"
        prompt = "Explain how robots perceive their environment"
        
        # Run the async method
        asyncio.run(client.send_prompt(model_name, prompt))
        
        # Keep node running to process callbacks
        rclpy.spin(client)
        
    except KeyboardInterrupt:
        print('\nCaught KeyboardInterrupt! Shutting down...')
    finally:
        # Clean up resources
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()