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
    bool goToHomePosition();
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

    int best_index = 10;

    

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    rclcpp::Client<pose_estimation_interfaces::srv::PoseEstimation>::SharedPtr pose_estimation_client_;
    rclcpp::Client<stack_optimization_interfaces::srv::StackOptimizer>::SharedPtr stack_optimizer_client_;

    rclcpp::Client<vg_control_interfaces::srv::VacuumSet>::SharedPtr vacuum_set_client_;
    rclcpp::Client<vg_control_interfaces::srv::VacuumRelease>::SharedPtr vacuum_release_client_;


    const std::string arm_group_name_ = "ur_arm";
    const std::string tcp_frame_ = "ur10_tcp";
    const double HALF_HOME_POSITION_X = -0.353;

    const double HOME_POSITION_X = -0.433;
    const double HOME_POSITION_Y = 0.641;
    const double HOME_POSITION_Z = -0.406;
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

bool BoxProcessorNode::goToHomePosition() {
    RCLCPP_INFO(get_logger(), "Moving to home position [%f, %f, %f]...", 
                HOME_POSITION_X, HOME_POSITION_Y, HOME_POSITION_Z);
    
    // Get current pose
    geometry_msgs::msg::PoseStamped current_pose = move_group_->getCurrentPose();
    
    // Create the home position target
    geometry_msgs::msg::Pose home_pose = current_pose.pose;
    home_pose.position.x = HOME_POSITION_X;
    home_pose.position.y = HOME_POSITION_Y;
    home_pose.position.z = HOME_POSITION_Z;

    home_pose.orientation.x = 0.991;
    home_pose.orientation.y = 0.132;
    home_pose.orientation.z = -0.002;
    home_pose.orientation.w = -0.001;
    
    // Set reasonable velocity for home position motion
    move_group_->setMaxVelocityScalingFactor(0.05);
    move_group_->setMaxAccelerationScalingFactor(0.05);
    
    // Plan and move to home position
    move_group_->setPoseTarget(home_pose);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    moveit::core::MoveItErrorCode error_code = move_group_->plan(plan);
    bool success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
    
    if (!success) {
        RCLCPP_ERROR(get_logger(), "Failed to plan to home position");
        return false;
    }
    
    error_code = move_group_->execute(plan);
    success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);
    
    if (!success) {
        RCLCPP_ERROR(get_logger(), "Failed to move to home position");
        return false;
    }
    
    // Get current joint values to set wrist_3 joint to zero
    std::vector<double> joint_values = move_group_->getCurrentJointValues();
    
    // Find the index of wrist_3 joint
    const std::vector<std::string>& joint_names = move_group_->getJointNames();
    int wrist3_index = -1;
    for (size_t i = 0; i < joint_names.size(); i++) {
        if (joint_names[i] == "ur10_wrist_3_joint") {
            wrist3_index = i;
            break;
        }
    }
    
    if (wrist3_index != -1) {
        // Set wrist_3 joint to zero
        joint_values[wrist3_index] = 0.0;
        
        // Plan and execute joint space move
        move_group_->setJointValueTarget(joint_values);
        moveit::planning_interface::MoveGroupInterface::Plan joint_plan;
        success = (move_group_->plan(joint_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        
        if (success) {
            RCLCPP_INFO(get_logger(), "Executing wrist adjustment to zero at home position");
            success = (move_group_->execute(joint_plan) == moveit::core::MoveItErrorCode::SUCCESS);
            if (!success) {
                RCLCPP_WARN(get_logger(), "Failed to adjust wrist_3 joint at home position, continuing anyway");
                // Don't return false, just continue with the operation
            }
        } else {
            RCLCPP_WARN(get_logger(), "Failed to plan wrist_3 adjustment at home position, continuing anyway");
        }
    } else {
        RCLCPP_WARN(get_logger(), "Could not find ur10_wrist_3_joint in joint names, skipping adjustment");
    }
    
    RCLCPP_INFO(get_logger(), "Successfully moved to home position");
    return true;
}

bool BoxProcessorNode::detectBox(geometry_msgs::msg::PoseStamped& box_pose,
                              geometry_msgs::msg::Vector3& dimensions) {
    // Call PoseEstimation service
    auto request = std::make_shared<pose_estimation_interfaces::srv::PoseEstimation::Request>();
    request->text_prompt = "box";
    
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

    //Log current robot orientation
    RCLCPP_INFO(get_logger(), "Current robot orientation: [%f, %f, %f, %f]", 
        current_quat.w, current_quat.x, current_quat.y, current_quat.z);
    //Log response position and orientations
    RCLCPP_INFO(get_logger(), "Response position: [%f, %f, %f]",
        response->position.x, response->position.y, response->position.z);
    RCLCPP_INFO(get_logger(), "Response orientations: ");
    for (size_t i = 0; i < response->orientations.size(); i++) {
        RCLCPP_INFO(get_logger(), "Orientation %zu: [%f, %f, %f, %f]", 
            i, response->orientations[i].w, response->orientations[i].x, 
            response->orientations[i].y, response->orientations[i].z);
    }
    // Find the orientation with the highest dot product (smallest angle difference)
    
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

    // print the best orientation
    RCLCPP_INFO(get_logger(), "Best orientation: [%f, %f, %f, %f]", 
        response->orientations[best_index].w, response->orientations[best_index].x, 
        response->orientations[best_index].y, response->orientations[best_index].z);
    
    // Fill box pose with response data
    box_pose.header.frame_id = "ur10_base_link";
    box_pose.pose.position = response->position;
    box_pose.pose.orientation = response->orientations[best_index]; // Use best orientation
    
    // Fill dimensions from response (convert from cm to meters)
    dimensions.x = response->x_width / 100.0;
    dimensions.y = response->y_length / 100.0;
    dimensions.z = response->z_height / 100.0;
    box_pose.pose.position.z += 0.118;
    
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

    int place_holder_dimension_x = dimensions.x * 100.0; // Convert to cm
    int place_holder_dimension_y = dimensions.y * 100.0; // Convert to cm
    
    //RCLINFo palce_holder_dimensions before swapping 
    RCLCPP_INFO(get_logger(), "Place holder dimensions before swapping: [%d, %d]", place_holder_dimension_x, place_holder_dimension_y);
    
    // Swap dimensions based on the best index
    if (best_index == 1 || best_index == 3) {
        int temp = place_holder_dimension_x;
        place_holder_dimension_x = place_holder_dimension_y;
        place_holder_dimension_y = temp;
        // Log the swapped dimensions
        RCLCPP_INFO(get_logger(), "Swapped place holder dimensions: [%d, %d]", place_holder_dimension_x, place_holder_dimension_y);
    }

    request->width = static_cast<float>(place_holder_dimension_x);
    request->length = static_cast<float>(place_holder_dimension_y);
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
    //Log response position and orientation
    RCLCPP_INFO(get_logger(), "Response position: [%f, %f, %f]",
        response->position.x, response->position.y, response->position.z);
    RCLCPP_INFO(get_logger(), "Response orientation: [%f, %f, %f, %f]", 
        response->orientations[0].w, response->orientations[0].x, 
        response->orientations[0].y, response->orientations[0].z);
    
    // Get current robot orientation - this line was missing
    geometry_msgs::msg::PoseStamped current_pose = move_group_->getCurrentPose();
    auto current_quat = current_pose.pose.orientation;
    int new_best_index = 0;
    double highest_dot_product = -1.0;

    // Define fixed candidate quaternions
    std::vector<geometry_msgs::msg::Quaternion> candidate_quats(4);
        
    // 0° [−0.7153, 0.6988, 0.0000, 0.0000]
    candidate_quats[0].x = -0.7153;
    candidate_quats[0].y = 0.6988;
    candidate_quats[0].z = 0.0000;
    candidate_quats[0].w = 0.0000;
    
    // 90° [−0.0117, 0.9999, 0.0000, 0.0000]
    candidate_quats[1].x = -0.0117;
    candidate_quats[1].y = 0.9999;
    candidate_quats[1].z = 0.0000;
    candidate_quats[1].w = 0.0000;
    
    // 180° [0.6988, 0.7153, 0.0000, 0.0000]
    candidate_quats[2].x = 0.6988;
    candidate_quats[2].y = 0.7153;
    candidate_quats[2].z = 0.0000;
    candidate_quats[2].w = 0.0000;
    
    // 270° [0.9999, 0.0117, 0.0000, 0.0000]
    candidate_quats[3].x = 0.9999;
    candidate_quats[3].y = 0.0117;
    candidate_quats[3].z = 0.0000;
    candidate_quats[3].w = 0.0000;
    

    // Check which of the two possible quaternions we have
    bool is_identity_rotation = std::abs(response->orientations[0].w - 1.0) < 0.01 && 
                        std::abs(response->orientations[0].x) < 0.01 && 
                        std::abs(response->orientations[0].y) < 0.01 && 
                        std::abs(response->orientations[0].z) < 0.01;

    bool is_90_degree_rotation = std::abs(response->orientations[0].w - 0.7071) < 0.01 && 
                        std::abs(response->orientations[0].x) < 0.01 && 
                        std::abs(response->orientations[0].y) < 0.01 && 
                        std::abs(response->orientations[0].z - 0.7071) < 0.01;

    
    if (is_identity_rotation) {
        // We only consider orientations 0 and 2 (0° and 180°)
        
        
        // Only check indices 0 and 2 (0° and 180°)
        for (int i : {0, 2}) {
            auto& candidate_quat = candidate_quats[i];
            
            // Calculate dot product between quaternions
            double dot_product = std::abs(
                current_quat.w * candidate_quat.w +
                current_quat.x * candidate_quat.x +
                current_quat.y * candidate_quat.y +
                current_quat.z * candidate_quat.z
            );
            
            if (dot_product > highest_dot_product) {
                highest_dot_product = dot_product;
                new_best_index = i;
            }
        }
    } 
    // Case 2: Response quaternion is [0.7071, 0.0000, 0.0000, 0.7071]
    else if (is_90_degree_rotation) {
        // Define fixed candidate quaternions
        
        
        // Only check indices 1 and 3 (90° and 270°)
        for (int i : {1, 3}) {
            auto& candidate_quat = candidate_quats[i];
            
            // Calculate dot product between quaternions
            double dot_product = std::abs(
                current_quat.w * candidate_quat.w +
                current_quat.x * candidate_quat.x +
                current_quat.y * candidate_quat.y +
                current_quat.z * candidate_quat.z
            );
            
            if (dot_product > highest_dot_product) {
                highest_dot_product = dot_product;
                new_best_index = i;
            }
        }
    } 
    // Fallback: If neither of the expected quaternions, use standard comparison
    else {
        // Define fixed candidate quaternions
        
        
        // Check all orientations
        for (size_t i = 0; i < candidate_quats.size(); i++) {
            auto& candidate_quat = candidate_quats[i];
            
            // Calculate dot product between quaternions
            double dot_product = std::abs(
                current_quat.w * candidate_quat.w +
                current_quat.x * candidate_quat.x +
                current_quat.y * candidate_quat.y +
                current_quat.z * candidate_quat.z
            );
            
            if (dot_product > highest_dot_product) {
                highest_dot_product = dot_product;
                new_best_index = i;
            }
        }
    }

    RCLCPP_INFO(get_logger(), "Selected orientation %d with dot product %f", 
                new_best_index, highest_dot_product);
    // Log the best orientation
     RCLCPP_INFO(get_logger(), "Best orientation: [%f, %f, %f, %f]", 
        candidate_quats[new_best_index].w, candidate_quats[new_best_index].x, 
        candidate_quats[new_best_index].y, candidate_quats[new_best_index].z);



    place_pose.pose.orientation = candidate_quats[new_best_index]; // Use first orientation
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

  // Activate vacuum gripper
  auto vacuum_request = std::make_shared<vg_control_interfaces::srv::VacuumSet::Request>();
  vacuum_request->channel_a = 210; // vacuum
  vacuum_request->channel_b = 210; // vacuum
  
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
  
  
  //setWristConstraint();
  // Set faster velocity for lift motion
  move_group_->setMaxVelocityScalingFactor(0.0075);
  move_group_->setMaxAccelerationScalingFactor(0.0075);
  
  // Lift the object (20cm up)
  waypoints.clear();
  geometry_msgs::msg::Pose lift_pose = pick_pose.pose;
  lift_pose.position.z += 0.4;
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

  // Get current joint values
  std::vector<double> joint_values = move_group_->getCurrentJointValues();
  
  // Find the index of wrist_3 joint
  const std::vector<std::string>& joint_names = move_group_->getJointNames();
  int wrist3_index = -1;
  for (size_t i = 0; i < joint_names.size(); i++) {
      if (joint_names[i] == "ur10_wrist_3_joint") {
          wrist3_index = i;
          break;
      }
  }
  
  if (wrist3_index != -1) {
      // Set wrist_3 joint to zero
      joint_values[wrist3_index] = 0.0;
      
      // Set slower speed for joint adjustment
      move_group_->setMaxVelocityScalingFactor(0.05);
      move_group_->setMaxAccelerationScalingFactor(0.05);
      
      // Plan and execute joint space move
      move_group_->setJointValueTarget(joint_values);
      moveit::planning_interface::MoveGroupInterface::Plan joint_plan;
      success = (move_group_->plan(joint_plan) == moveit::core::MoveItErrorCode::SUCCESS);
      
      if (success) {
          RCLCPP_INFO(get_logger(), "Executing wrist adjustment plan");
          success = (move_group_->execute(joint_plan) == moveit::core::MoveItErrorCode::SUCCESS);
          if (!success) {
              RCLCPP_WARN(get_logger(), "Failed to adjust wrist_3 joint, continuing anyway");
              // Don't return false, just continue with the operation
          }
      } else {
          RCLCPP_WARN(get_logger(), "Failed to plan wrist_3 adjustment, continuing anyway");
      }
  } else {
      RCLCPP_WARN(get_logger(), "Could not find ur10_wrist_3_joint in joint names, skipping adjustment");
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
   target_pose.pose.position.z += 0.15; // Adjust height for placement
   
   geometry_msgs::msg::Pose pre_place_pose = target_pose.pose;
   pre_place_pose.position.z += 0.1;
   
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
  waypoints.push_back(target_pose.pose);
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


  waypoints.clear();
  geometry_msgs::msg::Pose safety_pose = pre_place_pose;
  safety_pose.position.x = 0.819;
  safety_pose.position.y = 0.884;
  // Keep same z-height and orientation as pre_place_pose
  
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

  std::vector<double> joint_values = move_group_->getCurrentJointValues();
  
  // Find the index of wrist_3 joint
  const std::vector<std::string>& joint_names = move_group_->getJointNames();
  int wrist3_index = -1;
  for (size_t i = 0; i < joint_names.size(); i++) {
      if (joint_names[i] == "ur10_wrist_3_joint") {
          wrist3_index = i;
          break;
      }
  }
  
  if (wrist3_index != -1) {
      // Set wrist_3 joint to zero
      joint_values[wrist3_index] = 0.0;
      
      // Set slower speed for joint adjustment
      move_group_->setMaxVelocityScalingFactor(0.05);
      move_group_->setMaxAccelerationScalingFactor(0.05);
      
      // Plan and execute joint space move
      move_group_->setJointValueTarget(joint_values);
      moveit::planning_interface::MoveGroupInterface::Plan joint_plan;
      success = (move_group_->plan(joint_plan) == moveit::core::MoveItErrorCode::SUCCESS);
      
      if (success) {
          RCLCPP_INFO(get_logger(), "Executing wrist adjustment plan before home position");
          success = (move_group_->execute(joint_plan) == moveit::core::MoveItErrorCode::SUCCESS);
          if (!success) {
              RCLCPP_WARN(get_logger(), "Failed to adjust wrist_3 joint before home, continuing anyway");
              // Don't return false, just continue with the operation
          }
      } else {
          RCLCPP_WARN(get_logger(), "Failed to plan wrist_3 adjustment before home, continuing anyway");
      }
  } else {
      RCLCPP_WARN(get_logger(), "Could not find ur10_wrist_3_joint in joint names, skipping adjustment");
  }

  // After reaching the safety waypoint, go back to home position
  RCLCPP_INFO(get_logger(), "Moving back to home position");
  if (!goToHomePosition()) {
      RCLCPP_ERROR(get_logger(), "Failed to return to home position after placing");
      // Not returning false here, as we've already placed the box successfully
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

    if (!goToHomePosition()) {
        RCLCPP_ERROR(get_logger(), "Failed to move to home position. Aborting box processing.");
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
