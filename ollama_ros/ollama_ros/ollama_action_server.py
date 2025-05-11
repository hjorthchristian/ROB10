import re
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
import ollama
from ollama_ros_interfaces.action import ProcessThinking

class DepalletizerLLMServer(Node):
    def __init__(self):
        super().__init__('depalletizer_llm_server')
        self._action_server = ActionServer(
            self,
            ProcessThinking,
            'process_thinking',
            self.execute_callback
        )
        
        # System roles for each LLM
        self.worker_system_role = """You are an assistant for a depalletizing operation. When given a command, respond only with the description of the item(s).
Format multiple items by separating them with periods (.). Always end with a period. Remember to always be consistent throuout your answers. 
Examples:
- Command: "pick up the red box" → Response: "red box."
- Command: "take me the big package" → Response: "big package."
- Command: "give me the small jacket and the blue box." → Response: "small jacket. blue box."
- Command: "get me the box" → Response: "box."
"""
        
        self.validator_system_role = """You are a validator for a depalletizing robot's item recognition system. 
Check if the user's command specifies object(s) to pick up, and if the extracted items match what was requested.

If no objects were specified in the command, respond with: "NO_OBJECTS_SPECIFIED: Please specify which objects you want me to pick up."
If the extraction is correct, respond with: "VALID: true"
If the extraction is incorrect, respond with: "VALID: false" followed by an explanation.
You should never say valid if the items have not been extracer or if there are no items to extract it can olny and only be successful if the items have been extracted.
"""
        
        self.get_logger().info('Depalletizer LLM Server is ready')

    async def execute_callback(self, goal_handle):
        user_input = goal_handle.request.prompt
        model_name = goal_handle.request.model_name
        
        self.get_logger().info(f'Processing depalletizing command: "{user_input}" with model {model_name}')
        
        # Setup feedback and result objects
        feedback_msg = ProcessThinking.Feedback()
        result = ProcessThinking.Result()
        
        try:
            # Step 1: Worker LLM extracts items
            self.get_logger().info('Step 1: Extracting items with worker LLM')
            worker_messages = [
                {"role": "system", "content": self.worker_system_role},
                {"role": "user", "content": user_input}
            ]
            
            worker_response = ollama.chat(
                model=model_name,
                messages=worker_messages
            )
            extracted_items = worker_response["message"]["content"]
            self.get_logger().info(f'Extracted items: "{extracted_items}"')
            
            # Send feedback about item extraction (without success flag)
            feedback_msg.partial_content = f"Extracted items: {extracted_items}"
            feedback_msg.is_thinking = True  
            # No longer set feedback_msg.succes here
            goal_handle.publish_feedback(feedback_msg)
            
            # Step 2: Validator LLM validates extraction
            self.get_logger().info('Step 2: Validating extraction with validator LLM')
            validator_prompt = f"""User command: "{user_input}"
    Extracted items: "{extracted_items}"
    """
            
            validator_messages = [
                {"role": "system", "content": self.validator_system_role},
                {"role": "user", "content": validator_prompt}
            ]
            
            validator_response = ollama.chat(
                model=model_name,
                messages=validator_messages
            )
            validation_result = validator_response["message"]["content"]
            self.get_logger().info(f'Validation result: "{validation_result}"')
            
            # Send feedback about validation (without success flag)
            feedback_msg.partial_content = f"Validation result: {validation_result}"
            feedback_msg.is_thinking = True
            goal_handle.publish_feedback(feedback_msg)
            
            # Process validation result
            success = "VALID: true" in validation_result
            
            # Determine the final result based on validation
            if "NO_OBJECTS_SPECIFIED" in validation_result:
                result.answer = validation_result
                final_success = False
            else:
                result.answer = extracted_items if success else validation_result
                final_success = success
            
            # Set success in the result instead of feedback
            result.success = final_success
            
            # Send final feedback (without success flag)
            feedback_msg.is_thinking = False
            # No longer set feedback_msg.succes here
            goal_handle.publish_feedback(feedback_msg)
            
            # Return result with both answer and success status
            self.get_logger().info(f'Final result: "{result.answer}" (Success: {final_success})')
            goal_handle.succeed()
            return result
                
        except Exception as e:
            self.get_logger().error(f'Error processing request: {str(e)}')
            result.answer = f"ERROR: {str(e)}"
            result.success = False  # Set success to false on error
            goal_handle.succeed()
            return result

def main(args=None):
    rclpy.init(args=args)
    server = DepalletizerLLMServer()
    rclpy.spin(server)
    server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
