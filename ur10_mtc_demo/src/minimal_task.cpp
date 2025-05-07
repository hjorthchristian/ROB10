#include <rclcpp/rclcpp.hpp>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include "pose_estimation_interfaces/srv/pose_estimation.hpp"
#include "stack_optimization_interfaces/srv/stack_optimizer.hpp"
#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
#include <tf2_eigen/tf2_eigen.hpp>
#else
#include <tf2_eigen/tf2_eigen.h>
#endif

static const rclcpp::Logger LOGGER = rclcpp::get_logger("ur10_mtc_stacked_box_processor");
namespace mtc = moveit::task_constructor;

class MTCTaskNode
{
public:
  MTCTaskNode(const rclcpp::NodeOptions& options);

  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr getNodeBaseInterface();

  void processBoxes();

private:
  // Detect box using PoseEstimation service
  bool detectBox(geometry_msgs::msg::PoseStamped& box_pose, geometry_msgs::msg::Vector3& dimensions);
  
  // Get optimal placement using StackOptimizer service
  bool getPlacementPose(const geometry_msgs::msg::Vector3& dimensions,
                      const geometry_msgs::msg::Quaternion& orientation,
                      geometry_msgs::msg::PoseStamped& place_pose);
  
  // Create task for picking a box
  mtc::Task createPickTask(const geometry_msgs::msg::PoseStamped& pick_pose, 
                          const geometry_msgs::msg::Vector3& dimensions,
                          const std::string& object_id);
  
  // Create task for placing a box
  mtc::Task createPlaceTask(const geometry_msgs::msg::PoseStamped& place_pose,
                           const std::string& object_id);
  
  // Execute a task
  bool executeTask(mtc::Task& task);

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
  rclcpp::Client<pose_estimation_interfaces::srv::PoseEstimation>::SharedPtr pose_estimation_client_;
  rclcpp::Client<stack_optimization_interfaces::srv::StackOptimizer>::SharedPtr stack_optimizer_client_;
};

MTCTaskNode::MTCTaskNode(const rclcpp::NodeOptions& options)
  : node_{ std::make_shared<rclcpp::Node>("stacked_box_processor_node", options) }
{
  // Initialize planning scene interface
  planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
  
  // Initialize service clients
  pose_estimation_client_ = node_->create_client<pose_estimation_interfaces::srv::PoseEstimation>("estimate_pose");
  stack_optimizer_client_ = node_->create_client<stack_optimization_interfaces::srv::StackOptimizer>("box_stack_optimizer");
  
  // Wait for services to be available
  if (!pose_estimation_client_->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(LOGGER, "PoseEstimation service not available");
  }
  
  if (!stack_optimizer_client_->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(LOGGER, "StackOptimizer service not available");
  }
}

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr MTCTaskNode::getNodeBaseInterface()
{
  return node_->get_node_base_interface();
}

void MTCTaskNode::processBoxes()
{
  if (!pose_estimation_client_->service_is_ready()) {
    RCLCPP_ERROR(LOGGER, "PoseEstimation service is not available. Cannot process boxes.");
    return;
  }
    
  if (!stack_optimizer_client_->service_is_ready()) {
    RCLCPP_ERROR(LOGGER, "StackOptimizer service is not available. Cannot process boxes.");
    return;
  }
  
  int box_count = 0;
  bool continue_processing = true;
  
  while (continue_processing && rclcpp::ok()) {
    RCLCPP_INFO(LOGGER, "Processing box #%d", box_count + 1);
    
    // 1. Detect the box
    geometry_msgs::msg::PoseStamped box_pose;
    geometry_msgs::msg::Vector3 dimensions;
    
    if (!detectBox(box_pose, dimensions)) {
      RCLCPP_INFO(LOGGER, "No more boxes detected. Finishing...");
      break;
    }
    
    std::string object_id = "box_" + std::to_string(box_count);
    
    // 2. Create and execute pick task
    auto pick_task = createPickTask(box_pose, dimensions, object_id);
    if (!executeTask(pick_task)) {
      RCLCPP_ERROR(LOGGER, "Failed to pick box #%d", box_count + 1);
      continue_processing = false;
      break;
    }
    
    // 3. Get optimal placement pose
    geometry_msgs::msg::PoseStamped place_pose;
    if (!getPlacementPose(dimensions, box_pose.pose.orientation, place_pose)) {
      RCLCPP_ERROR(LOGGER, "Failed to get placement pose for box #%d", box_count + 1);
      continue_processing = false;
      break;
    }
    
    // 4. Create and execute place task
    auto place_task = createPlaceTask(place_pose, object_id);
    if (!executeTask(place_task)) {
      RCLCPP_ERROR(LOGGER, "Failed to place box #%d", box_count + 1);
      continue_processing = false;
      break;
    }
    
    RCLCPP_INFO(LOGGER, "Successfully processed box #%d", box_count + 1);
    box_count++;
    
    // Wait briefly before processing the next box
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }
  
  RCLCPP_INFO(LOGGER, "Finished processing. Total boxes handled: %d", box_count);
}

bool MTCTaskNode::detectBox(geometry_msgs::msg::PoseStamped& box_pose, 
                           geometry_msgs::msg::Vector3& dimensions)
{
  // Call PoseEstimation service
  auto request = std::make_shared<pose_estimation_interfaces::srv::PoseEstimation::Request>();
  request->text_prompt = "box in top middle";
  
  auto future = pose_estimation_client_->async_send_request(request);
  
  if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
    RCLCPP_ERROR(LOGGER, "PoseEstimation service request timed out");
    return false;
  }
  
  auto response = future.get();
  
  if (!response->success) {
    RCLCPP_ERROR(LOGGER, "PoseEstimation service failed: %s", response->error_message.c_str());
    return false;
  }
  
  // If no orientations are provided, we have no valid box
  if (response->orientations.empty()) {
    RCLCPP_ERROR(LOGGER, "No orientation provided by PoseEstimation service");
    return false;
  }
  
  // Fill box pose with response data
  box_pose.header.frame_id = "ur10_base_link";  // Ensure consistent frame
  box_pose.pose.position = response->position;
  box_pose.pose.orientation = response->orientations[0]; // Use first orientation
  
  // Fill dimensions from response (convert from cm to meters)
  dimensions.x = response->x_width / 100.0;
  dimensions.y = response->y_length / 100.0;
  dimensions.z = response->z_height / 100.0;
  
  RCLCPP_INFO(LOGGER, "Detected box at position: [%f, %f, %f] with dimensions: [%f, %f, %f]", 
             box_pose.pose.position.x, box_pose.pose.position.y, box_pose.pose.position.z,
             dimensions.x, dimensions.y, dimensions.z);
  
  return true;
}

bool MTCTaskNode::getPlacementPose(const geometry_msgs::msg::Vector3& dimensions,
                                 const geometry_msgs::msg::Quaternion& orientation,
                                 geometry_msgs::msg::PoseStamped& place_pose)
{
  // Call StackOptimizer service
  auto request = std::make_shared<stack_optimization_interfaces::srv::StackOptimizer::Request>();
  request->width = static_cast<int32_t>(dimensions.x * 100.0);  // Convert to cm
  request->length = static_cast<int32_t>(dimensions.y * 100.0);
  request->height = static_cast<int32_t>(dimensions.z * 100.0);
  request->orientation = orientation;
  request->change_stack_allowed = false;  // Don't rearrange existing stack
  
  auto future = stack_optimizer_client_->async_send_request(request);
  
  if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
    RCLCPP_ERROR(LOGGER, "StackOptimizer service request timed out");
    return false;
  }
  
  auto response = future.get();
  
  if (!response->success) {
    RCLCPP_ERROR(LOGGER, "StackOptimizer service failed");
    return false;
  }
  
  // If no orientations are provided, we have no valid placement
  if (response->orientations.empty()) {
    RCLCPP_ERROR(LOGGER, "No orientation provided by StackOptimizer service");
    return false;
  }
  
  // Fill placement pose with response data
  place_pose.header.frame_id = "ur10_base_link";  // Ensure consistent frame
  place_pose.pose.position = response->position;
  place_pose.pose.orientation = response->orientations[0]; // Use first orientation
  
  RCLCPP_INFO(LOGGER, "Place position determined: [%f, %f, %f]", 
             place_pose.pose.position.x, place_pose.pose.position.y, place_pose.pose.position.z);
  
  return true;
}

mtc::Task MTCTaskNode::createPickTask(const geometry_msgs::msg::PoseStamped& pick_pose,
    const geometry_msgs::msg::Vector3& dimensions,
    const std::string& object_id)
{
mtc::Task task;
task.stages()->setName("Pick Box Task");
task.loadRobotModel(node_);

const auto& arm_group_name = "ur_arm";
const auto& tcp_frame = "ur10_tcp";

// Set task properties
task.setProperty("group", arm_group_name);
task.setProperty("ik_frame", tcp_frame);
task.setProperty("reference_frame", "ur10_base_link");  // Add this line

// Configure cartesian planner with improved parameters
auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
cartesian_planner->setMaxVelocityScalingFactor(0.1);  // Increased from 0.01
cartesian_planner->setMaxAccelerationScalingFactor(0.1);  // Increased from 0.01
cartesian_planner->setStepSize(0.01);  // Increased from 0.005
cartesian_planner->setMinFraction(0.5);  // Set min_fraction on the planner, not the stage

// Set precision parameters
moveit::core::CartesianPrecision precision;
precision.rotational = 0.01;  // Angular precision constraint
precision.translational = 0.01;  // Positional precision constraint
cartesian_planner->setPrecision(precision);

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

RCLCPP_INFO(LOGGER, "Added box '%s' to planning scene", object_id.c_str());

// 1. Current state
auto stage_current_state = std::make_unique<mtc::stages::CurrentState>("current state");
task.add(std::move(stage_current_state));

// 2. Move to pre-grasp position - CORRECTED DIRECTION
auto pre_grasp_pose = pick_pose;
pre_grasp_pose.pose.position.z += 0.1; // 10cm ABOVE the object
pre_grasp_pose.header.frame_id = "ur10_base_link";  // Ensure frame is set

auto stage_approach = std::make_unique<mtc::stages::MoveTo>("approach", cartesian_planner);
stage_approach->setGroup(arm_group_name);
stage_approach->setGoal(pre_grasp_pose);
stage_approach->setIKFrame(tcp_frame);
stage_approach->properties().set("enforce_cartesian_motion", true);  // Force Cartesian
task.add(std::move(stage_approach));

// 3. Move down to grasp position
auto stage_grasp = std::make_unique<mtc::stages::MoveTo>("grasp", cartesian_planner);
stage_grasp->setGroup(arm_group_name);
stage_grasp->setGoal(pick_pose);
stage_grasp->setIKFrame(tcp_frame);
stage_grasp->properties().set("enforce_cartesian_motion", true);  // Force Cartesian
task.add(std::move(stage_grasp));

// 4. Attach object to the robot
auto stage_attach = std::make_unique<mtc::stages::ModifyPlanningScene>("attach object");
stage_attach->attachObject(object_id, tcp_frame);
task.add(std::move(stage_attach));

// 5. Lift the object - CORRECTED DIRECTION
auto lift_pose = pick_pose;
lift_pose.pose.position.z += 0.1; // 10cm UP
lift_pose.header.frame_id = "ur10_base_link";  // Ensure frame is set

auto stage_lift = std::make_unique<mtc::stages::MoveTo>("lift", cartesian_planner);
stage_lift->setGroup(arm_group_name);
stage_lift->setGoal(lift_pose);
stage_lift->setIKFrame(tcp_frame);
stage_lift->properties().set("enforce_cartesian_motion", true);  // Force Cartesian
task.add(std::move(stage_lift));

return task;
}


mtc::Task MTCTaskNode::createPlaceTask(const geometry_msgs::msg::PoseStamped& place_pose,
    const std::string& object_id)
{
mtc::Task task;
task.stages()->setName("Place Box Task");
task.loadRobotModel(node_);

const auto& arm_group_name = "ur_arm";
const auto& tcp_frame = "ur10_tcp";

// Set task properties
task.setProperty("group", arm_group_name);
task.setProperty("ik_frame", tcp_frame);
task.setProperty("reference_frame", "ur10_base_link");  // Add this line

// Configure cartesian planner with improved parameters
auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
cartesian_planner->setMaxVelocityScalingFactor(0.1);  // Increased from 0.01
cartesian_planner->setMaxAccelerationScalingFactor(0.1);  // Increased from 0.01
cartesian_planner->setStepSize(0.01);  // Increased from 0.005
cartesian_planner->setMinFraction(0.5);  // Set min_fraction on the planner, not the stage

// Set precision parameters
moveit::core::CartesianPrecision precision;
precision.rotational = 0.01;  // Angular precision constraint
precision.translational = 0.01;  // Positional precision constraint
cartesian_planner->setPrecision(precision);

// 1. Current state
auto stage_current_state = std::make_unique<mtc::stages::CurrentState>("current state");
task.add(std::move(stage_current_state));

// 2. Move to pre-place position - CORRECTED DIRECTION
auto pre_place_pose = place_pose;
pre_place_pose.pose.position.z += 0.1; // 10cm ABOVE the place position
pre_place_pose.header.frame_id = "ur10_base_link";  // Ensure frame is set

auto stage_pre_place = std::make_unique<mtc::stages::MoveTo>("pre-place", cartesian_planner);
stage_pre_place->setGroup(arm_group_name);
stage_pre_place->setGoal(pre_place_pose);
stage_pre_place->setIKFrame(tcp_frame);
stage_pre_place->properties().set("enforce_cartesian_motion", true);  // Force Cartesian
task.add(std::move(stage_pre_place));

// 3. Move down to place position
auto stage_place = std::make_unique<mtc::stages::MoveTo>("place", cartesian_planner);
stage_place->setGroup(arm_group_name);
stage_place->setGoal(place_pose);
stage_place->setIKFrame(tcp_frame);
stage_place->properties().set("enforce_cartesian_motion", true);  // Force Cartesian
task.add(std::move(stage_place));

// 4. Detach the object
auto stage_detach = std::make_unique<mtc::stages::ModifyPlanningScene>("detach object");
stage_detach->detachObject(object_id, tcp_frame);
task.add(std::move(stage_detach));

// 5. Retreat - CORRECTED DIRECTION
auto stage_retreat = std::make_unique<mtc::stages::MoveTo>("retreat", cartesian_planner);
stage_retreat->setGroup(arm_group_name);
stage_retreat->setGoal(pre_place_pose);
stage_retreat->setIKFrame(tcp_frame);
stage_retreat->properties().set("enforce_cartesian_motion", true);  // Force Cartesian
task.add(std::move(stage_retreat));

return task;
}


bool MTCTaskNode::executeTask(mtc::Task& task)
{
  try {
    task.init();
  }
  catch (mtc::InitStageException& e) {
    RCLCPP_ERROR_STREAM(LOGGER, "Task initialization failed: " << e);
    return false;
  }
  
  if (!task.plan(5)) {
    RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
    return false;
  }
  
  RCLCPP_INFO(LOGGER, "Task planning succeeded");
  task.introspection().publishSolution(*task.solutions().front());
  
  auto result = task.execute(*task.solutions().front());
  if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
    RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
    return false;
  }
  
  RCLCPP_INFO(LOGGER, "Task execution completed successfully");
  return true;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);

  auto mtc_task_node = std::make_shared<MTCTaskNode>(options);
  rclcpp::executors::MultiThreadedExecutor executor;

  auto spin_thread = std::make_unique<std::thread>([&executor, &mtc_task_node]() {
    executor.add_node(mtc_task_node->getNodeBaseInterface());
    executor.spin();
    executor.remove_node(mtc_task_node->getNodeBaseInterface());
  });

  mtc_task_node->processBoxes();

  spin_thread->join();
  rclcpp::shutdown();
  return 0;
}
