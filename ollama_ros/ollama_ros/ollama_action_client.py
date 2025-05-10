import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import asyncio
from ollama_ros_interfaces.action import ProcessThinking

class Qwen3Client(Node):
    def __init__(self):
        super().__init__('qwen3_client')
        self._client = ActionClient(self, ProcessThinking, 'process_thinking')
        
    async def send_prompt(self, model_name, prompt):
        # Wait for server
        self._client.wait_for_server()
        
        # Create goal
        goal_msg = ProcessThinking.Goal()
        goal_msg.model_name = model_name
        goal_msg.prompt = prompt
        
        # Send goal with feedback callback
        self._send_goal_future = self._client.send_goal_async(
            goal_msg, 
            feedback_callback=self._feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self._goal_response_callback)
        
    def _feedback_callback(self, feedback_msg):
        if feedback_msg.feedback.is_thinking:
            # Process thinking feedback (could log, display, etc.)
            self.get_logger().info(f"Thinking: {feedback_msg.feedback.partial_content}")
        else:
            # Process regular content
            self.get_logger().info(f"Responding: {feedback_msg.feedback.partial_content}")
    
    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
            
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_callback)
        
    def _get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Final answer: {result.answer}')
def main(args=None):
    rclpy.init(args=args)
    client = Qwen3Client()

    
    # Define example prompt
    model_name = "qwen3:1.7b"
    prompt = "Explain how robots perceive their environment"
    
    # Run the async method
    asyncio.run(client.send_prompt(model_name, prompt))
    
    # Keep node running to process callbacks
    rclpy.spin(client)
    
    # Clean up resources
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
