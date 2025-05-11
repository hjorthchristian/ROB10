import re
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
import ollama

from ollama_ros_interfaces.action import ProcessThinking

class Qwen3ThinkingServer(Node):
    def __init__(self):
        super().__init__('qwen3_thinking_server')
        self._action_server = ActionServer(
            self,
            ProcessThinking,
            'process_thinking',
            self.execute_callback
        )
        self.get_logger().info('Qwen3 Thinking Server is ready')

    async def execute_callback(self, goal_handle):
        self.get_logger().info(f'Processing prompt with model {goal_handle.request.model_name}')
        
        # Setup feedback and result
        feedback_msg = ProcessThinking.Feedback()
        result = ProcessThinking.Result()
        
        # Start streaming from Ollama
        stream = ollama.chat(
            model=goal_handle.request.model_name,
            messages=[{"role": "user", "content": goal_handle.request.prompt}],
            stream=True
        )
        
        # Variables to track content
        full_content = ""
        thinking_content = ""
        final_answer = ""
        in_thinking = False
        
        # Process streaming response
        for chunk in stream:
            content = chunk['message']['content']
            full_content += content
            
            # Check for thinking tags
            if "<think>" in content:
                in_thinking = True
            
            if "</think>" in content:
                in_thinking = False
            
            # Publish as feedback with thinking flag
            feedback_msg.partial_content = content
            feedback_msg.is_thinking = in_thinking
            goal_handle.publish_feedback(feedback_msg)
        
        # Extract final answer (remove thinking section)
        final_answer = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
        
        # Return only the final answer
        result.answer = final_answer
        goal_handle.succeed()
        
        return result
    
def main(args=None):
    rclpy.init(args=args)
    server = Qwen3ThinkingServer()
    rclpy.spin(server)
    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
