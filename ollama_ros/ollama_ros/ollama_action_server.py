import re
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
import ollama
from ollama_ros_interfaces.action import ProcessThinking  # Updated import path

class DepalletizerLLMServer(Node):

    def __init__(self):
        super().__init__('depalletizer_llm_server')
        self._action_server = ActionServer(
            self,
            ProcessThinking,
            'process_thinking',
            self.execute_callback)

        # System roles for each LLM
        self.worker_system_role = """You are an assistant for a depalletizing operation. When given a command, respond only with the description of the item(s).
        Format multiple items by separating them with periods (.). Always end with a period. Remember to always be consistent throuout your answers. Also like the last
        example, if the user uses a plural word, you should use the singular form of the word in your answer.
        Examples:
        - Command: "pick up the red box" → Response: "red box."
        - Command: "take me the big package" → Response: "big package."
        - Command: "give me the small jacket and the blue box." → Response: "small jacket. blue box."
        - Command: "get me the box" → Response: "box."
        - Command: "get me all of the boxes" → Response: "box."
        """

        self.validator_system_role = """You are a validator for a depalletizing robot's item recognition system.
        Check if the user's command specifies object(s) to pick up, and if the extracted items match what was requested.
        If no objects were specified in the command, respond with: "NO_OBJECTS_SPECIFIED: Please specify which objects you want me to pick up."
        If the extraction is correct, respond with: "VALID: true"
        If the extraction is incorrect, respond with: "VALID: false" followed by an explanation.
        You should never say valid if the items have not been extracer or if there are no items to extract it can olny and only be successful if the items have been extracted.
        Look at the other system prompt's example so you also can validate it:
         Examples:
        - Command: "pick up the red box" → Response: "red box."
        - Command: "take me the big package" → Response: "big package."
        - Command: "give me the small jacket and the blue box." → Response: "small jacket. blue box."
        - Command: "get me the box" → Response: "box."
        - Command: "get me all of the boxes" → Response: "box."
        So you can use the above ecamples to help in the validation process.
        """

        self.get_logger().info('Depalletizer LLM Server is ready')

    def _clean_response(self, response):
        """Remove <think> tags and their content from responses"""
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    async def execute_callback(self, goal_handle):
        user_input = goal_handle.request.prompt
        model_name = "qwen3:1.7b"

        self.get_logger().info(f'Processing depalletizing command: "{user_input}" with model {model_name}')

        feedback_msg = ProcessThinking.Feedback()
        result = ProcessThinking.Result()

        try:
            # Step 1: Extract items
            self.get_logger().info('Step 1: Extracting items')
            worker_messages = [
                {"role": "system", "content": self.worker_system_role},
                {"role": "user", "content": user_input}
            ]

            worker_response = ollama.chat(model=model_name, messages=worker_messages)
            extracted_items = worker_response["message"]["content"]

            self.get_logger().info('Items Extracted')

            
            # Send raw feedback with thinking tags
            feedback_msg.partial_content = f"Extracted items: {extracted_items}"
            feedback_msg.is_thinking = True
            goal_handle.publish_feedback(feedback_msg)

            # Step 2: Validate extraction
            self.get_logger().info('Step 2: Validating extraction')
            validator_prompt = f"User command: {user_input}\nExtracted items: {extracted_items}"
            validator_messages = [
                {"role": "system", "content": self.validator_system_role},
                {"role": "user", "content": validator_prompt}
            ]

            validator_response = ollama.chat(model=model_name, messages=validator_messages)
            validation_result = validator_response["message"]["content"]

            # Send raw validation feedback
            feedback_msg.partial_content = f"Validation result: {validation_result}"
            goal_handle.publish_feedback(feedback_msg)

            # Clean responses for final result
            cleaned_items = self._clean_response(extracted_items)
            cleaned_validation = self._clean_response(validation_result)

            # Prepare final result
            success = "VALID: true" in validation_result
            if "NO_OBJECTS_SPECIFIED" in validation_result:
                result.answer = cleaned_validation
                result.reasoning = "No objects specified in command"
            else:
                result.answer = cleaned_items if success else cleaned_validation
                result.reasoning = f"Validation: {cleaned_validation}"

            result.success = success

            # Final feedback without thinking flag  
            feedback_msg.is_thinking = False
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info(f'Final result: {result.answer}')
            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
            result.answer = f"ERROR: {str(e)}"
            result.reasoning = f"Exception: {str(e)}"
            result.success = False
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
