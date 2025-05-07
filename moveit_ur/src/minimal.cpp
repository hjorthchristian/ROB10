#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include "pose_estimation_interfaces/srv/pose_estimation.hpp"
#include "stack_optimization_interfaces/srv/stack_optimizer.hpp"
#include <thread>

class BoxProcessorNode : public rclcpp::Node
{
public:
  BoxProcessorNode(const std::string &node_name);
  // Initialize MoveGroupInterface after node construction
  void initialize();
  // Factory method for proper instantiation
  static std::shared_ptr<BoxProcessorNode> create(const std::string &node_name);
  void processBoxes();

private:
  // Detect box using PoseEstimation service
  bool detectBox(geometry_msgs::msg::PoseStamped& box_pose, geometry_msgs::msg::Vector3& dimensions);
  // Get optimal placement using StackOptimizer service
  bool getPlacementPose(const geometry_msgs::msg::Vector3& dimensions,
                        const geometry_msgs::msg::Quaternion& orientation,
                        geometry_msgs::msg::PoseStamped& place_pose);
  // Pick a box
  bool pickBox(const geometry_msgs::msg::PoseStamped& pick_pose,
              const geometry_msgs::msg::Vector3& dimensions,
              const std::string& object_id);
  // Place a box
  bool placeBox(const geometry_msgs::msg::PoseStamped& place_pose,
               const std::string& object_id);
  // Check if position is feasible for planning
  bool isPositionFeasible(const geometry_msgs::msg::Pose& pose);
  // Move to home position
  bool moveToHome();

  // Class members
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
  rclcpp::Client<pose_estimation_interfaces::srv::PoseEstimation>::SharedPtr pose_estimation_client_;
  rclcpp::Client<stack_optimization_interfaces::srv::StackOptimizer>::SharedPtr stack_optimizer_client_;
  const std::string arm_group_name_ = "ur_arm";
  const std::string tcp_frame_ = "ur10_tcp";
};

BoxProcessorNode::BoxProcessorNode(const std::string &node_name)
  : Node(node_name)
{
  // Initialize service clients only in constructor
  pose_estimation_client_ = create_client<pose_estimation_interfaces::srv::PoseEstimation>("estimate_pose");
  stack_optimizer_client_ = create_client<stack_optimization_interfaces::srv::StackOptimizer>("box_stack_optimizer");
  // Initialize planning scene interface
  planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
}

void BoxProcessorNode::initialize()
{
  // Initialize MoveGroupInterface - now safe to use shared_from_this()
  move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), arm_group_name_);
      
  // Set planning reference frame to ur10_base_link
  move_group_->setPoseReferenceFrame("ur10_base_link");
  
  // Set planning parameters with improved values
  move_group_->setMaxVelocityScalingFactor(0.2);  // Increased from 0.1
  move_group_->setMaxAccelerationScalingFactor(0.2);  // Increased from 0.1
  move_group_->setPlanningTime(10.0);  // Increased from 5.0
  move_group_->setNumPlanningAttempts(20);  // Increased from 10
  move_group_->setGoalPositionTolerance(0.02);  // Increased from 0.01
  move_group_->setGoalOrientationTolerance(0.03);  // Increased from 0.01
  
  // Wait for services to be available
  if (!pose_estimation_client_->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(get_logger(), "PoseEstimation service not available");
  }
  
  if (!stack_optimizer_client_->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(get_logger(), "StackOptimizer service not available");
  }
}

std::shared_ptr<BoxProcessorNode> BoxProcessorNode::create(const std::string &node_name)
{
  auto node = std::make_shared<BoxProcessorNode>(node_name);
  node->initialize();
  return node;
}

bool BoxProcessorNode::isPositionFeasible(const geometry_msgs::msg::Pose& pose) {
  // Check if the Z position is too low - adjust threshold based on your robot
  if (pose.position.z < -0.5) {
    RCLCPP_WARN(get_logger(), "Target position may be too low (z = %f)", pose.position.z);
    return false;
  }
  
  // Add more checks as needed - could check position is within workspace
  
  return true;
}

bool BoxProcessorNode::moveToHome()
{
  RCLCPP_INFO(get_logger(), "Moving to home position...");
  
  move_group_->setNamedTarget("home");
  
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  moveit::core::MoveItErrorCode error_code = move_group_->plan(plan);
  bool success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
  
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to plan to home position, error code: %d", error_code.val);
    return false;
  }
  
  error_code = move_group_->execute(plan);
  success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
  
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to move to home position, error code: %d", error_code.val);
    return false;
  }
  
  RCLCPP_INFO(get_logger(), "Successfully moved to home position");
  return true;
}

bool BoxProcessorNode::detectBox(geometry_msgs::msg::PoseStamped& box_pose,
                                geometry_msgs::msg::Vector3& dimensions)
{
  // Call PoseEstimation service
  auto request = std::make_shared<pose_estimation_interfaces::srv::PoseEstimation::Request>();
  request->text_prompt = "box in top middle";
  auto future = pose_estimation_client_->async_send_request(request);
  
  if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
    RCLCPP_ERROR(get_logger(), "PoseEstimation service request timed out");
    return false;
  }
  
  auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(get_logger(), "PoseEstimation service failed: %s", response->error_message.c_str());
    return false;
  }
  
  // If no orientations are provided, we have no valid box
  if (response->orientations.empty()) {
    RCLCPP_ERROR(get_logger(), "No orientation provided by PoseEstimation service");
    return false;
  }
  
  // Fill box pose with response data
  box_pose.header.frame_id = "ur10_base_link";
  box_pose.pose.position = response->position;
  box_pose.pose.orientation = response->orientations[0]; // Use first orientation
  
  // Fill dimensions from response (convert from cm to meters)
  dimensions.x = response->x_width / 100.0;
  dimensions.y = response->y_length / 100.0;
  dimensions.z = response->z_height / 100.0;
  
  RCLCPP_INFO(get_logger(), "Detected box at position: [%f, %f, %f] with dimensions: [%f, %f, %f]",
              box_pose.pose.position.x, box_pose.pose.position.y, box_pose.pose.position.z,
              dimensions.x, dimensions.y, dimensions.z);
              
  // Check if the position is feasible
  if (!isPositionFeasible(box_pose.pose)) {
    RCLCPP_ERROR(get_logger(), "Detected box position is not feasible for planning");
    return false;
  }
  
  return true;
}

bool BoxProcessorNode::getPlacementPose(const geometry_msgs::msg::Vector3& dimensions,
                                       const geometry_msgs::msg::Quaternion& orientation,
                                       geometry_msgs::msg::PoseStamped& place_pose)
{
  // Call StackOptimizer service
  auto request = std::make_shared<stack_optimization_interfaces::srv::StackOptimizer::Request>();
  request->width = static_cast<int>(dimensions.x * 100.0); // Convert to cm
  request->length = static_cast<int>(dimensions.y * 100.0);
  request->height = static_cast<int>(dimensions.z * 100.0);
  request->orientation = orientation;
  request->change_stack_allowed = false; // Don't rearrange existing stack
  
  auto future = stack_optimizer_client_->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
    RCLCPP_ERROR(get_logger(), "StackOptimizer service request timed out");
    return false;
  }
  
  auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(get_logger(), "StackOptimizer service failed");
    return false;
  }
  
  // If no orientations are provided, we have no valid placement
  if (response->orientations.empty()) {
    RCLCPP_ERROR(get_logger(), "No orientation provided by StackOptimizer service");
    return false;
  }
  
  // Fill placement pose with response data
  place_pose.header.frame_id = "ur10_base_link";
  place_pose.pose.position = response->position;
  place_pose.pose.orientation = response->orientations[0]; // Use first orientation
  
  RCLCPP_INFO(get_logger(), "Place position determined: [%f, %f, %f]",
              place_pose.pose.position.x, place_pose.pose.position.y, place_pose.pose.position.z);
  
  return true;
}

bool BoxProcessorNode::pickBox(const geometry_msgs::msg::PoseStamped& pick_pose,
                              const geometry_msgs::msg::Vector3& dimensions,
                              const std::string& object_id)
{
  // Add box to planning scene
  moveit_msgs::msg::CollisionObject collision_object;
  collision_object.header = pick_pose.header;
  collision_object.id = object_id;
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = primitive.BOX;
  primitive.dimensions.resize(3);
  primitive.dimensions[0] = dimensions.x;
  primitive.dimensions[1] = dimensions.y;
  primitive.dimensions[2] = dimensions.z;
  collision_object.primitives.push_back(primitive);
  collision_object.primitive_poses.push_back(pick_pose.pose);
  collision_object.operation = collision_object.ADD;
  std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
  collision_objects.push_back(collision_object);
  planning_scene_interface_->addCollisionObjects(collision_objects);
  RCLCPP_INFO(get_logger(), "Added box '%s' to planning scene", object_id.c_str());
  
  // Wait for planning scene to update
  rclcpp::sleep_for(std::chrono::seconds(1));
  
  // First move to a known good position (home or similar)
 
  
  // Create pre-grasp pose - 20cm above the object (increased from 10cm)
  geometry_msgs::msg::Pose pre_grasp_pose = pick_pose.pose;
  pre_grasp_pose.position.z += 0.3;  // Increased from 0.1
  
  // Log the target pose for debugging
  RCLCPP_INFO(get_logger(), "Planning to pre-grasp pose: [%f, %f, %f] with orientation [%f, %f, %f, %f]",
    pre_grasp_pose.position.x, pre_grasp_pose.position.y, pre_grasp_pose.position.z,
    pre_grasp_pose.orientation.x, pre_grasp_pose.orientation.y, 
    pre_grasp_pose.orientation.z, pre_grasp_pose.orientation.w);
  
  // Use these improved planning parameters for this specific motion
  move_group_->setPlanningTime(10.0);  
  move_group_->setNumPlanningAttempts(20);
  
  // Plan and move to pre-grasp pose
  move_group_->setPoseTarget(pre_grasp_pose);
  
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  moveit::core::MoveItErrorCode error_code = move_group_->plan(plan);
  bool success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
  
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to plan to pre-grasp position, error code: %d", error_code.val);
    return false;
  }
  
  error_code = move_group_->execute(plan);
  success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to move to pre-grasp position, error code: %d", error_code.val);
    return false;
  }
  
  // Plan and execute Cartesian path to grasp pose with more generous parameters
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(pick_pose.pose);
  moveit_msgs::msg::RobotTrajectory trajectory;
  double eef_step = 0.02;  // 2cm step instead of 1cm
  double jump_threshold = 0.0;  // disable jump threshold
  
  double fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
  if (fraction < 0.8) {  // Accept 80% completion instead of 90%
    RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path to grasp position (%.2f%% achieved)", fraction * 100.0);
    return false;
  }
  
  plan.trajectory = trajectory;
  success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to move to grasp position");
    return false;
  }
  
  // Attach the object
  move_group_->attachObject(object_id, tcp_frame_);
  RCLCPP_INFO(get_logger(), "Attached object '%s'", object_id.c_str());
  
  // Wait for planning scene to update
  rclcpp::sleep_for(std::chrono::seconds(1));
  
  // Lift the object (20cm up instead of 10cm)
  waypoints.clear();
  geometry_msgs::msg::Pose lift_pose = pick_pose.pose;
  lift_pose.position.z += 0.2;  // Increased from 0.1
  waypoints.push_back(lift_pose);
  
  fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
  if (fraction < 0.8) {  // Accept 80% completion
    RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path for lifting (%.2f%% achieved)", fraction * 100.0);
    return false;
  }
  
  plan.trajectory = trajectory;
  success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to lift object");
    return false;
  }
  
  RCLCPP_INFO(get_logger(), "Successfully picked up object '%s'", object_id.c_str());
  return true;
}

bool BoxProcessorNode::placeBox(const geometry_msgs::msg::PoseStamped& place_pose,
                               const std::string& object_id)
{
  // Create pre-place pose - 20cm above the place position
  geometry_msgs::msg::Pose pre_place_pose = place_pose.pose;
  pre_place_pose.position.z += 0.2;  // Increased from 0.1
  
  // Plan and move to pre-place pose
  move_group_->setPoseTarget(pre_place_pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  
  moveit::core::MoveItErrorCode error_code = move_group_->plan(plan);
  bool success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
  
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to plan to pre-place position, error code: %d", error_code.val);
    return false;
  }
  
  error_code = move_group_->execute(plan);
  success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to move to pre-place position, error code: %d", error_code.val);
    return false;
  }
  
  // Plan and execute Cartesian path to place pose
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(place_pose.pose);
  moveit_msgs::msg::RobotTrajectory trajectory;
  double eef_step = 0.02;  // 2cm step
  double jump_threshold = 0.0;  // disable jump threshold
  
  double fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
  if (fraction < 0.8) {  // Accept 80% completion
    RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path to place position (%.2f%% achieved)", fraction * 100.0);
    return false;
  }
  
  plan.trajectory = trajectory;
  success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to move to place position");
    return false;
  }
  
  // Detach the object
  move_group_->detachObject(object_id);
  RCLCPP_INFO(get_logger(), "Detached object '%s'", object_id.c_str());
  
  // Wait for planning scene to update
  rclcpp::sleep_for(std::chrono::seconds(1));
  
  // Retreat (20cm up)
  waypoints.clear();
  waypoints.push_back(pre_place_pose);
  
  fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
  if (fraction < 0.8) {  // Accept 80% completion
    RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path for retreat (%.2f%% achieved)", fraction * 100.0);
    return false;
  }
  
  plan.trajectory = trajectory;
  success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Failed to retreat");
    return false;
  }
  
  RCLCPP_INFO(get_logger(), "Successfully placed object '%s'", object_id.c_str());
  return true;
}

void BoxProcessorNode::processBoxes()
{
  if (!pose_estimation_client_->service_is_ready()) {
    RCLCPP_ERROR(get_logger(), "PoseEstimation service is not available. Cannot process boxes.");
    return;
  }
  
  if (!stack_optimizer_client_->service_is_ready()) {
    RCLCPP_ERROR(get_logger(), "StackOptimizer service is not available. Cannot process boxes.");
    return;
  }
  
  // Start with a move to home position
  if (!moveToHome()) {
    RCLCPP_ERROR(get_logger(), "Failed to move to initial home position");
    return;
  }
  
  int box_count = 0;
  bool continue_processing = true;
  
  while (continue_processing && rclcpp::ok()) {
    RCLCPP_INFO(get_logger(), "Processing box #%d", box_count + 1);
    
    // 1. Detect the box
    geometry_msgs::msg::PoseStamped box_pose;
    geometry_msgs::msg::Vector3 dimensions;
    if (!detectBox(box_pose, dimensions)) {
      RCLCPP_INFO(get_logger(), "No more boxes detected. Finishing...");
      break;
    }
    
    std::string object_id = "box_" + std::to_string(box_count);
    
    // 2. Pick the box
    if (!pickBox(box_pose, dimensions, object_id)) {
      RCLCPP_ERROR(get_logger(), "Failed to pick box #%d", box_count + 1);
      continue_processing = false;
      break;
    }
    
    // 3. Get optimal placement pose
    geometry_msgs::msg::PoseStamped place_pose;
    if (!getPlacementPose(dimensions, box_pose.pose.orientation, place_pose)) {
      RCLCPP_ERROR(get_logger(), "Failed to get placement pose for box #%d", box_count + 1);
      continue_processing = false;
      break;
    }
    
    // 4. Place the box
    if (!placeBox(place_pose, object_id)) {
      RCLCPP_ERROR(get_logger(), "Failed to place box #%d", box_count + 1);
      continue_processing = false;
      break;
    }
    
    RCLCPP_INFO(get_logger(), "Successfully processed box #%d", box_count + 1);
    box_count++;
    
    // Wait briefly before processing the next box
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }
  
  RCLCPP_INFO(get_logger(), "Finished processing. Total boxes handled: %d", box_count);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = BoxProcessorNode::create("move_to_pose_movegroup");
  
  // Spin in a separate thread
  std::thread spin_thread([node]() {
    rclcpp::spin(node);
  });
  
  // Process boxes
  node->processBoxes();
  
  // Clean up
  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
