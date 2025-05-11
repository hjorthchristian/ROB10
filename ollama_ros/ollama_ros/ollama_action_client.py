import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import asyncio
import time
import signal
import sys
import threading
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
        self._processing_done = None  # Track when processing completes
        self._is_processing = False   # Flag to prevent multiple simultaneous requests
        
    async def send_prompt(self, model_name, prompt):
        # Create future to track completion
        self._processing_done = asyncio.Future()
        self._is_processing = True
        
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
        
        # Wait for processing to complete
        await self._processing_done
        self._is_processing = False
        
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
            if self._processing_done and not self._processing_done.done():
                self._processing_done.set_result(None)
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
        
        # Check for success flag in the result
        if hasattr(result, 'success') and result.success:
            self.get_logger().info('Operation was successful')
        else:
            self.get_logger().warn('Operation failed or no success flag present')
        
        self.get_logger().info(f'Final answer: {result.answer}')
        
        # Signal that processing is complete
        if self._processing_done and not self._processing_done.done():
            self._processing_done.set_result(None)

# Function to handle interactive session
async def interactive_session(client, model_name):
    print(f"\nInteractive session with {model_name}")
    print("Type 'exit' or 'quit' to end the session")
    print("---------------------------------------------")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                print("Ending session")
                break
                
            # Process the prompt
            await client.send_prompt(model_name, user_input)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error processing prompt: {e}")

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
        # Create executor for handling callbacks
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(client)
        
        # Run the executor in a separate thread
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        # Start interactive session
        model_name = "qwen3:1.7b"
        asyncio.run(interactive_session(client, model_name))
        
    except KeyboardInterrupt:
        print('\nCaught KeyboardInterrupt! Shutting down...')
    finally:
        # Clean up resources
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
