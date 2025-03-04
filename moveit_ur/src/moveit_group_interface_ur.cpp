/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, SRI International
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of SRI International nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Sachin Chitta, Dave Coleman, Mike Lautman */

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>

#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>

#include <moveit_visual_tools/moveit_visual_tools.h>

// All source files that use ROS logging should define a file-specific
// static const rclcpp::Logger named LOGGER, located at the top of the file
// and inside the namespace with the narrowest scope (if there is one)
static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_demo");

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto move_group_node = rclcpp::Node::make_shared("move_group_interface_ur", node_options);

  // We spin up a SingleThreadedExecutor for the current state monitor to get information
  // about the robot's state.
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(move_group_node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // BEGIN_TUTORIAL
  //
  // Setup
  // ^^^^^
  //
  // MoveIt operates on sets of joints called "planning groups" and stores them in an object called
  // the ``JointModelGroup``. Throughout MoveIt, the terms "planning group" and "joint model group"
  // are used interchangeably.
  static const std::string PLANNING_GROUP = "ur_manipulator";

  // The
  // :moveit_codedir:`MoveGroupInterface<moveit_ros/planning_interface/move_group_interface/include/moveit/move_group_interface/move_group_interface.h>`
  // class can be easily set up using just the name of the planning group you would like to control and plan for.
  moveit::planning_interface::MoveGroupInterface move_group(move_group_node, PLANNING_GROUP);

  // We will use the
  // :moveit_codedir:`PlanningSceneInterface<moveit_ros/planning_interface/planning_scene_interface/include/moveit/planning_scene_interface/planning_scene_interface.h>`
  // class to add and remove collision objects in our "virtual world" scene
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  // Raw pointers are frequently used to refer to the planning group for improved performance.
  const moveit::core::JointModelGroup* joint_model_group =
      move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

  // Visualization
  // ^^^^^^^^^^^^^
  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools(move_group_node, "base_link", "move_group_tutorial",
                                                      move_group.getRobotModel());

  visual_tools.deleteAllMarkers();

  /* Remote control is an introspection tool that allows users to step through a high level script */
  /* via buttons and keyboard shortcuts in RViz */
  visual_tools.loadRemoteControl();

  // RViz provides many types of markers, in this demo we will use text, cylinders, and spheres
  Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
  text_pose.translation().z() = 1.0;
  visual_tools.publishText(text_pose, "MoveGroupInterface_Demo", rvt::WHITE, rvt::XLARGE);

  // Batch publishing is used to reduce the number of messages being sent to RViz for large visualizations
  visual_tools.trigger();

  // Getting Basic Information
  // ^^^^^^^^^^^^^^^^^^^^^^^^^
  //
  // We can print the name of the reference frame for this robot.
  RCLCPP_INFO(LOGGER, "Planning frame: %s", move_group.getPlanningFrame().c_str());


  // We can get a list of all the groups in the robot:
  RCLCPP_INFO(LOGGER, "Available Planning Groups:");
  std::copy(move_group.getJointModelGroupNames().begin(), move_group.getJointModelGroupNames().end(),
            std::ostream_iterator<std::string>(std::cout, ", "));

  // Start the demo
  // ^^^^^^^^^^^^^^^^^^^^^^^^^
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");

  // .. _move_group_interface-planning-to-pose-goal:
  //

  // Now, we call the planner to compute the plan and visualize it.
  // Note that we are just planning, not asking move_group
  // to actually move the robot.
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;

  bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

  RCLCPP_INFO(LOGGER, "Visualizing plan 1 (pose goal) %s", success ? "" : "FAILED");

// Get the current pose of the end effector
geometry_msgs::msg::PoseStamped current_pose = move_group.getCurrentPose();
RCLCPP_INFO(LOGGER, "Current end effector pose: x=%.3f, y=%.3f, z=%.3f",
            current_pose.pose.position.x, 
            current_pose.pose.position.y, 
            current_pose.pose.position.z);

// Create a waypoint vector for the Cartesian path
std::vector<geometry_msgs::msg::Pose> waypoints;

// Add the current pose as the starting waypoint (optional but helpful for visualization)
waypoints.push_back(current_pose.pose);

// Create a target pose that's 5cm higher in the Z direction
geometry_msgs::msg::Pose target_pose = current_pose.pose;
target_pose.position.z -= 0.05;  // Add 5cm in Z axis
target_pose.position.y -= 0.05;  // Add 5cm in Y axis
RCLCPP_INFO(LOGGER, "Target end effector pose: x=%.3f, y=%.3f, z=%.3f",
             target_pose.position.x, 
             target_pose.position.y, 
             target_pose.position.z);

waypoints.push_back(target_pose);

move_group.setMaxVelocityScalingFactor(0.125);  // 25% of maximum velocity
move_group.setMaxAccelerationScalingFactor(0.125);  // 25% of maximum acceleration

// Compute the Cartesian path
moveit_msgs::msg::RobotTrajectory trajectory;
const double jump_threshold = 2.0;  // Disable jump threshold
const double eef_step = 0.01;       // 1cm resolution for trajectory
double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

RCLCPP_INFO(LOGGER, "Cartesian path (%.2f%% achieved)", fraction * 100.0);

RCLCPP_INFO(LOGGER, "Trajectory has %ld points", trajectory.joint_trajectory.points.size());
if (!trajectory.joint_trajectory.points.empty()) {
  RCLCPP_INFO(LOGGER, "First trajectory point time: %.3f", 
              trajectory.joint_trajectory.points.front().time_from_start.sec + 
              trajectory.joint_trajectory.points.front().time_from_start.nanosec/1e9);
  RCLCPP_INFO(LOGGER, "Last trajectory point time: %.3f", 
              trajectory.joint_trajectory.points.back().time_from_start.sec + 
              trajectory.joint_trajectory.points.back().time_from_start.nanosec/1e9);
}

// Print the start and target poses for verification
RCLCPP_INFO(LOGGER, "Start pose: x=%.3f, y=%.3f, z=%.3f", 
            waypoints[0].position.x, waypoints[0].position.y, waypoints[0].position.z);
RCLCPP_INFO(LOGGER, "Target pose: x=%.3f, y=%.3f, z=%.3f", 
            waypoints[1].position.x, waypoints[1].position.y, waypoints[1].position.z);

// Print trajectory start and end positions (first and last point)
if (!trajectory.joint_trajectory.points.empty()) {
  auto& first_point = trajectory.joint_trajectory.points.front();
  auto& last_point = trajectory.joint_trajectory.points.back();
  
  RCLCPP_INFO(LOGGER, "Trajectory start positions:");
  for (size_t i = 0; i < first_point.positions.size(); ++i) {
    RCLCPP_INFO(LOGGER, "  Joint %ld: %.3f", i, first_point.positions[i]);
  }
  
  RCLCPP_INFO(LOGGER, "Trajectory end positions:");
  for (size_t i = 0; i < last_point.positions.size(); ++i) {
    RCLCPP_INFO(LOGGER, "  Joint %ld: %.3f", i, last_point.positions[i]);
  }
}

// Visualize the plan in RViz
visual_tools.deleteAllMarkers();
visual_tools.publishText(text_pose, "Move Up 5cm", rvt::WHITE, rvt::XLARGE);
visual_tools.publishPath(waypoints, rvt::LIME_GREEN, rvt::SMALL);
for (std::size_t i = 0; i < waypoints.size(); ++i)
    visual_tools.publishAxisLabeled(waypoints[i], "pt" + std::to_string(i), rvt::SMALL);
visual_tools.trigger();

// Execute the Cartesian trajectory if desired
if (fraction > 0.95) {  // Only execute if we achieved at least 95% of the path
    move_group.execute(trajectory);
}
  // The result may look like this:
  //
  // .. image:: ./move_group_interface_tutorial_clear_path.gif
  //    :alt: animation showing the arm moving relatively straight toward the goal
  //
  // Now, let's define a collision object ROS message for the robot to avoid.
//   moveit_msgs::msg::CollisionObject collision_object;
//   collision_object.header.frame_id = move_group.getPlanningFrame();

//   // The id of the object is used to identify it.
//   collision_object.id = "box1";

//   // Define a box to add to the world.
//   shape_msgs::msg::SolidPrimitive primitive;
//   primitive.type = primitive.BOX;
//   primitive.dimensions.resize(3);
//   primitive.dimensions[primitive.BOX_X] = 0.1;
//   primitive.dimensions[primitive.BOX_Y] = 1.5;
//   primitive.dimensions[primitive.BOX_Z] = 0.5;

//   // Define a pose for the box (specified relative to frame_id).
//   geometry_msgs::msg::Pose box_pose;
//   box_pose.orientation.w = 1.0;
//   box_pose.position.x = 0.48;
//   box_pose.position.y = 0.0;
//   box_pose.position.z = 0.25;

//   collision_object.primitives.push_back(primitive);
//   collision_object.primitive_poses.push_back(box_pose);
//   collision_object.operation = collision_object.ADD;

//   std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
//   collision_objects.push_back(collision_object);

//   // Now, let's add the collision object into the world
//   // (using a vector that could contain additional objects)
//   RCLCPP_INFO(LOGGER, "Add an object into the world");
//   planning_scene_interface.addCollisionObjects(collision_objects);

//   // Show text in RViz of status and wait for MoveGroup to receive and process the collision object message
//   visual_tools.publishText(text_pose, "Add_object", rvt::WHITE, rvt::XLARGE);
//   visual_tools.trigger();
//   visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to once the collision object appears in RViz");

//   // Now, when we plan a trajectory it will avoid the obstacle.
//   success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
//   RCLCPP_INFO(LOGGER, "Visualizing plan 6 (pose goal move around cuboid) %s", success ? "" : "FAILED");
//   visual_tools.publishText(text_pose, "Obstacle_Goal", rvt::WHITE, rvt::XLARGE);
//   visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
//   visual_tools.trigger();
//   visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window once the plan is complete");

  rclcpp::shutdown();
  return 0;
}