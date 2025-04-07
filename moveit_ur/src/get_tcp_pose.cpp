#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

int main(int argc, char** argv) {
  // Initialize ROS 2 node
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("get_end_effector_pose");

  // Create a TF2 buffer and listener to get transforms directly
  std::shared_ptr<tf2_ros::Buffer> tf_buffer = 
    std::make_shared<tf2_ros::Buffer>(node->get_clock());
  std::shared_ptr<tf2_ros::TransformListener> tf_listener = 
    std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  // Wait a bit for TF data to be available
  rclcpp::sleep_for(std::chrono::seconds(2));

  // Try to get the transform from base_link to end-effector
  geometry_msgs::msg::PoseStamped current_pose;
  bool transform_success = false;

  try {
    // First approach: Try to get the transform directly from TF2
    geometry_msgs::msg::TransformStamped transform = 
      tf_buffer->lookupTransform("ur10_base_link", "ur10_tcp", tf2::TimePointZero);
    
    // Convert the transform to a pose
    current_pose.header = transform.header;
    current_pose.pose.position.x = transform.transform.translation.x;
    current_pose.pose.position.y = transform.transform.translation.y;
    current_pose.pose.position.z = transform.transform.translation.z;
    current_pose.pose.orientation = transform.transform.rotation;
    
    transform_success = true;
    RCLCPP_INFO(node->get_logger(), "Successfully got transform from TF2");
  } 
  catch (const tf2::TransformException& ex) {
    RCLCPP_WARN(node->get_logger(), "Could not get transform from TF2: %s", ex.what());
    
    // Second approach: Try using MoveIt
    try {
      // Create MoveGroupInterface for the 'ur_arm' group
      moveit::planning_interface::MoveGroupInterface move_group(node, "ur_arm");
      
      // Set the reference frame to 'ur10_base_link'
      move_group.setPoseReferenceFrame("ur10_base_link");
      
      // Get the current pose of the end-effector link 'ur10_tcp'
      current_pose = move_group.getCurrentPose("ur10_tcp");
      transform_success = true;
      RCLCPP_INFO(node->get_logger(), "Successfully got pose from MoveIt");
    } 
    catch (const std::exception& e) {
      RCLCPP_ERROR(node->get_logger(), "Failed to get pose from MoveIt: %s", e.what());
    }
  }

  // Log the pose using ROS info logger
  if (transform_success) {
    RCLCPP_INFO(node->get_logger(), "End-effector pose in 'ur10_base_link':");
    RCLCPP_INFO(node->get_logger(), "Position: [x: %f, y: %f, z: %f]", 
                current_pose.pose.position.x, 
                current_pose.pose.position.y, 
                current_pose.pose.position.z);
    RCLCPP_INFO(node->get_logger(), "Orientation: [x: %f, y: %f, z: %f, w: %f]", 
                current_pose.pose.orientation.x, 
                current_pose.pose.orientation.y, 
                current_pose.pose.orientation.z, 
                current_pose.pose.orientation.w);
  } else {
    RCLCPP_ERROR(node->get_logger(), "Failed to get end-effector pose");
  }

  // Shutdown ROS 2
  rclcpp::shutdown();
  return 0;
}
