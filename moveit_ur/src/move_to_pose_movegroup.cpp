#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <geometric_shapes/shapes.h>
#include "pose_estimation_interfaces/srv/pose_estimation.hpp"
#include "stack_optimization_interfaces/srv/stack_optimizer.hpp"
#include <thread>
#include "vg_control_interfaces/srv/vacuum_set.hpp"
#include "vg_control_interfaces/srv/vacuum_release.hpp"

class BoxProcessorNode : public rclcpp::Node {
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
    // Simplified methods without object management
    bool pickBox(const geometry_msgs::msg::PoseStamped& pick_pose);
    bool placeBox(const geometry_msgs::msg::PoseStamped& place_pose);
    // Check if position is feasible for planning
    bool isPositionFeasible(const geometry_msgs::msg::Pose& pose);

    

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    rclcpp::Client<pose_estimation_interfaces::srv::PoseEstimation>::SharedPtr pose_estimation_client_;
    rclcpp::Client<stack_optimization_interfaces::srv::StackOptimizer>::SharedPtr stack_optimizer_client_;

    rclcpp::Client<vg_control_interfaces::srv::VacuumSet>::SharedPtr vacuum_set_client_;
    rclcpp::Client<vg_control_interfaces::srv::VacuumRelease>::SharedPtr vacuum_release_client_;


    const std::string arm_group_name_ = "ur_arm";
    const std::string tcp_frame_ = "ur10_tcp";
};

BoxProcessorNode::BoxProcessorNode(const std::string &node_name)
    : Node(node_name)
{
    // Initialize service clients only in constructor
    pose_estimation_client_ = create_client<pose_estimation_interfaces::srv::PoseEstimation>("estimate_pose");
    stack_optimizer_client_ = create_client<stack_optimization_interfaces::srv::StackOptimizer>("box_stack_optimizer");
    vacuum_set_client_ = create_client<vg_control_interfaces::srv::VacuumSet>("grip_adjust");
    vacuum_release_client_ = create_client<vg_control_interfaces::srv::VacuumRelease>("release_vacuum");

}


void BoxProcessorNode::initialize() {
    // Initialize MoveGroupInterface - now safe to use shared_from_this()
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), arm_group_name_);
    
    // Set planning reference frame to ur10_base_link
    move_group_->setPoseReferenceFrame("ur10_base_link");
    
    // Set planning parameters with improved values
    move_group_->setMaxVelocityScalingFactor(0.01);
    move_group_->setMaxAccelerationScalingFactor(0.01);
    move_group_->setPlanningTime(10.0);
    move_group_->setNumPlanningAttempts(20);
    move_group_->setGoalPositionTolerance(0.005);
    move_group_->setGoalOrientationTolerance(0.03);
    
    // Wait for services to be available
    if (!pose_estimation_client_->wait_for_service(std::chrono::seconds(5))) {
        RCLCPP_ERROR(get_logger(), "PoseEstimation service not available");
    }
    
    if (!stack_optimizer_client_->wait_for_service(std::chrono::seconds(5))) {
        RCLCPP_ERROR(get_logger(), "StackOptimizer service not available");
    }

    if (!vacuum_set_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "VacuumSet service not available");
    }
    else {
        RCLCPP_INFO(get_logger(), "VacuumSet service available");
    }
    if (!vacuum_release_client_->wait_for_service(std::chrono::seconds(5))) {
        RCLCPP_ERROR(get_logger(), "VacuumRelease service not available");
    }
  

}

std::shared_ptr<BoxProcessorNode> BoxProcessorNode::create(const std::string &node_name) {
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

bool BoxProcessorNode::detectBox(geometry_msgs::msg::PoseStamped& box_pose,
                              geometry_msgs::msg::Vector3& dimensions) {
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

    // Get current robot orientation
    geometry_msgs::msg::PoseStamped current_pose = move_group_->getCurrentPose();
    auto current_quat = current_pose.pose.orientation;
    
    // Find the orientation with the highest dot product (smallest angle difference)
    int best_index = 0;
    double highest_dot_product = -1.0; // Initialize to lowest possible value
    
    for (size_t i = 0; i < response->orientations.size(); i++) {
        auto& candidate_quat = response->orientations[i];
        
        // Calculate dot product between quaternions
        double dot_product = 
            current_quat.w * candidate_quat.w + 
            current_quat.x * candidate_quat.x + 
            current_quat.y * candidate_quat.y + 
            current_quat.z * candidate_quat.z;
            
        // Take absolute value since q and -q represent the same orientation
        dot_product = std::abs(dot_product);
        
        if (dot_product > highest_dot_product) {
            highest_dot_product = dot_product;
            best_index = i;
        }
    }
    
    RCLCPP_INFO(get_logger(), "Selected orientation %d with dot product %f", 
                best_index, highest_dot_product);
    
    // Fill box pose with response data
    box_pose.header.frame_id = "ur10_base_link";
    box_pose.pose.position = response->position;
    box_pose.pose.orientation = response->orientations[0]; // Use best orientation
    
    // Fill dimensions from response (convert from cm to meters)
    dimensions.x = response->x_width / 100.0;
    dimensions.y = response->y_length / 100.0;
    dimensions.z = response->z_height / 100.0;
    box_pose.pose.position.z += 0.11;
    
    RCLCPP_INFO(get_logger(), "Detected box at position: [%f, %f, %f] with dimensions: [%f, %f, %f]",
        box_pose.pose.position.x, box_pose.pose.position.y, box_pose.pose.position.z,
        dimensions.x, dimensions.y, dimensions.z);
    
    // Check if the position is feasible
    
    
    return true;
}

bool BoxProcessorNode::getPlacementPose(const geometry_msgs::msg::Vector3& dimensions,
                                      const geometry_msgs::msg::Quaternion& orientation,
                                      geometry_msgs::msg::PoseStamped& place_pose) {
    // Call StackOptimizer service
    auto request = std::make_shared<stack_optimization_interfaces::srv::StackOptimizer::Request>();
    request->width = static_cast<float>(dimensions.x * 100.0); // Convert to cm
    request->length = static_cast<float>(dimensions.y * 100.0);
    request->height = static_cast<float>(dimensions.z * 100.0);
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
    place_pose.pose.position.z += 0.12; // Adjust height for placement
    
    RCLCPP_INFO(get_logger(), "Place position determined: [%f, %f, %f]",
        place_pose.pose.position.x, place_pose.pose.position.y, place_pose.pose.position.z);
    return true;
}

bool BoxProcessorNode::pickBox(const geometry_msgs::msg::PoseStamped& pick_pose) {
  // Set faster velocity for pre-grasp motion
  move_group_->setMaxVelocityScalingFactor(0.05);
  move_group_->setMaxAccelerationScalingFactor(0.05);
  //setWristConstraint();

  // First move to a pre-grasp pose (30cm above the object)
  geometry_msgs::msg::Pose pre_grasp_pose = pick_pose.pose;
  pre_grasp_pose.position.z += 0.1;
  geometry_msgs::msg::PoseStamped current_robot_pose = move_group_->getCurrentPose();


  pre_grasp_pose.orientation = current_robot_pose.pose.orientation;
  
  // Log the target pose for debugging
  RCLCPP_INFO(get_logger(), "Planning to pre-grasp pose: [%f, %f, %f]...",
      pre_grasp_pose.position.x, pre_grasp_pose.position.y, pre_grasp_pose.position.z);
  
  // Plan and move to pre-grasp pose
  move_group_->setPoseTarget(pre_grasp_pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  moveit::core::MoveItErrorCode error_code = move_group_->plan(plan);
  bool success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
      RCLCPP_ERROR(get_logger(), "Failed to plan to pre-grasp position");
      //clearConstraints();
      return false;
  }

  
  error_code = move_group_->execute(plan);
  success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
      RCLCPP_ERROR(get_logger(), "Failed to move to pre-grasp position");
      //clearConstraints();
      return false;
  }
  //clearConstraints();
  
  // Slower velocity for actual grasp motion
  move_group_->setMaxVelocityScalingFactor(0.01);
  move_group_->setMaxAccelerationScalingFactor(0.01);
  
  // Plan and execute Cartesian path to grasp pose
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(pick_pose.pose);
  
  moveit_msgs::msg::RobotTrajectory trajectory;
  double eef_step = 0.005; // 5mm step
  double fraction = move_group_->computeCartesianPath(waypoints, eef_step, trajectory);
  if (fraction < 0.8) {
      RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path to grasp position");
      return false;
  }
  
  plan.trajectory = trajectory;
  success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
      RCLCPP_ERROR(get_logger(), "Failed to move to grasp position");
      return false;
  }
  
  // Activate vacuum gripper
  auto vacuum_request = std::make_shared<vg_control_interfaces::srv::VacuumSet::Request>();
  vacuum_request->channel_a = 150; // 60% vacuum
  vacuum_request->channel_b = 150; // 60% vacuum
  
  auto vacuum_future = vacuum_set_client_->async_send_request(vacuum_request);
  if (vacuum_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "VacuumSet service request timed out");
      return false;
  }
  
  auto vacuum_response = vacuum_future.get();
  if (!vacuum_response->success) {
      RCLCPP_ERROR(get_logger(), "VacuumSet service failed: %s", vacuum_response->message.c_str());
      return false;
  }
  
  // Wait for 2 seconds to ensure proper grip
  RCLCPP_INFO(get_logger(), "Waiting 2 seconds for vacuum grip to stabilize");
  rclcpp::sleep_for(std::chrono::seconds(2));
  //setWristConstraint();
  // Set faster velocity for lift motion
  move_group_->setMaxVelocityScalingFactor(0.0075);
  move_group_->setMaxAccelerationScalingFactor(0.0075);
  
  // Lift the object (20cm up)
  waypoints.clear();
  geometry_msgs::msg::Pose lift_pose = pick_pose.pose;
  lift_pose.position.z += 0.2;
  waypoints.push_back(lift_pose);
  fraction = move_group_->computeCartesianPath(waypoints, eef_step, trajectory);
  if (fraction < 0.8) {
      RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path for lifting");
      return false;
  }
  
  plan.trajectory = trajectory;
  success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
      RCLCPP_ERROR(get_logger(), "Failed to lift object");
      return false;
  }
  move_group_->setMaxVelocityScalingFactor(0.05);
  move_group_->setMaxAccelerationScalingFactor(0.05);
  waypoints.clear();
  geometry_msgs::msg::Pose safety_pose = lift_pose;
  safety_pose.position.x = 0.819;
  safety_pose.position.y = 0.884;
  // Keep same z-height and orientation as lift_pose
  
  RCLCPP_INFO(get_logger(), "Moving to safety waypoint at [%f, %f, %f]",
      safety_pose.position.x, safety_pose.position.y, safety_pose.position.z);
  
  waypoints.push_back(safety_pose);
  fraction = move_group_->computeCartesianPath(waypoints, eef_step, trajectory);
  if (fraction < 0.8) {
      RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path to safety waypoint");
      return false;
  }
  
  plan.trajectory = trajectory;
  success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
      RCLCPP_ERROR(get_logger(), "Failed to move to safety waypoint");
      return false;
  }

  //clearConstraints();

  RCLCPP_INFO(get_logger(), "Successfully completed pick trajectory");
  return true;
}



bool BoxProcessorNode::placeBox(const geometry_msgs::msg::PoseStamped& place_pose) {
  // Set faster velocity for pre-place motion
   // Set faster velocity for pre-place motion
   move_group_->setMaxVelocityScalingFactor(0.05);
   move_group_->setMaxAccelerationScalingFactor(0.05);
 
   // Get current pose to use as starting point for Cartesian path
   geometry_msgs::msg::PoseStamped current_pose = move_group_->getCurrentPose();
   
   geometry_msgs::msg::PoseStamped target_pose = place_pose;
   target_pose.pose.position.z += 0.30; // Adjust height for placement
   
   geometry_msgs::msg::Pose pre_place_pose = target_pose.pose;
   pre_place_pose.position.z += 0.2;
   
   // Plan and execute Cartesian path to pre-place pose
   std::vector<geometry_msgs::msg::Pose> waypoints;
   waypoints.push_back(pre_place_pose);
   
   moveit_msgs::msg::RobotTrajectory trajectory;
   double eef_step = 0.005; // 2cm step
   double fraction = move_group_->computeCartesianPath(waypoints, eef_step, trajectory);
   
   if (fraction < 0.8) {
       RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path to pre-place position (coverage: %f)", fraction);
       return false;
   }
   
   RCLCPP_INFO(get_logger(), "Moving to pre-place position with Cartesian path (coverage: %f)", fraction);
   
   // Create a plan from the trajectory and execute it
   moveit::planning_interface::MoveGroupInterface::Plan plan;
   plan.trajectory = trajectory;
   
   bool success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
   if (!success) {
       RCLCPP_ERROR(get_logger(), "Failed to move to pre-place position");
       return false;
   }
  
  // Slower velocity for actual place motion
  move_group_->setMaxVelocityScalingFactor(0.01);
  move_group_->setMaxAccelerationScalingFactor(0.01);
  
  // Plan and execute Cartesian path to place pose
  waypoints.clear();
  waypoints.push_back(place_pose.pose);
  fraction = move_group_->computeCartesianPath(waypoints, eef_step, trajectory);
  if (fraction < 0.8) {
      RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path to place position");
      return false;
  }
  
  plan.trajectory = trajectory;
  success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
      RCLCPP_ERROR(get_logger(), "Failed to move to place position");
      return false;
  }
  
  // Release vacuum
  auto release_request = std::make_shared<vg_control_interfaces::srv::VacuumRelease::Request>();
  release_request->release_vacuum = 1; // 1 to release vacuum
  
  auto release_future = vacuum_release_client_->async_send_request(release_request);
  if (release_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "VacuumRelease service request timed out");
      return false;
  }
  
  auto release_response = release_future.get();
  if (!release_response->success) {
      RCLCPP_ERROR(get_logger(), "VacuumRelease service failed: %s", release_response->message.c_str());
      return false;
  }
  
  // Set faster velocity for retreat motion
  move_group_->setMaxVelocityScalingFactor(0.05);
  move_group_->setMaxAccelerationScalingFactor(0.05);
  
  // Retreat (20cm up)
  waypoints.clear();
  waypoints.push_back(pre_place_pose);
  fraction = move_group_->computeCartesianPath(waypoints, eef_step, trajectory);
  if (fraction < 0.8) {
      RCLCPP_ERROR(get_logger(), "Failed to compute Cartesian path for retreat");
      return false;
  }
  
  plan.trajectory = trajectory;
  success = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!success) {
      RCLCPP_ERROR(get_logger(), "Failed to retreat");
      return false;
  }
  
  RCLCPP_INFO(get_logger(), "Successfully completed place trajectory");
  return true;
}


void BoxProcessorNode::processBoxes() {
    if (!pose_estimation_client_->service_is_ready()) {
        RCLCPP_ERROR(get_logger(), "PoseEstimation service is not available. Cannot process boxes.");
        return;
    }
    
    if (!stack_optimizer_client_->service_is_ready()) {
        RCLCPP_ERROR(get_logger(), "StackOptimizer service is not available. Cannot process boxes.");
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
        
        // 2. Pick the box (simplified - no object creation)
        if (!pickBox(box_pose)) {
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
        
        // 4. Place the box (simplified - no object management)
        if (!placeBox(place_pose)) {
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

int main(int argc, char **argv) {
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
