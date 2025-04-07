#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>

int main(int argc, char** argv) {
  // Initialize ROS 2 node
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("move_to_pose_movegroup");

  // Allow MoveGroup to receive information
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // Create MoveGroupInterface for the 'ur_arm' group
  moveit::planning_interface::MoveGroupInterface move_group(node, "ur_arm");
  
  // Set the reference frame to 'ur10_base_link'
  move_group.setPoseReferenceFrame("ur10_base_link");
  
  // Set the end effector link
  move_group.setEndEffectorLink("ur10_tcp");

  // Set planning parameters
  move_group.setPlanningTime(5.0);
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);
  
  // Create the target pose
  geometry_msgs::msg::Pose target_pose;
  target_pose.position.x = 0.734647;
  target_pose.position.y = 0.303083;
  target_pose.position.z = -0.436118;
  target_pose.orientation.x = -0.070557;
  target_pose.orientation.y = 0.997330;
  target_pose.orientation.z = 0.009466;
  target_pose.orientation.w = -0.016255;
  
  // Set the target pose
  move_group.setPoseTarget(target_pose);
  
  // Plan the motion
  RCLCPP_INFO(node->get_logger(), "Planning to target pose...");
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  
  if (success) {
    RCLCPP_INFO(node->get_logger(), "Planning successful! Executing motion...");
    move_group.execute(plan);
    RCLCPP_INFO(node->get_logger(), "Motion execution completed.");
  } else {
    RCLCPP_ERROR(node->get_logger(), "Planning failed!");
  }
  
  // Shutdown ROS 2
  rclcpp::shutdown();
  return 0;
}
