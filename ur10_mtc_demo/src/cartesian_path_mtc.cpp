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

static const rclcpp::Logger LOGGER = rclcpp::get_logger("ur10_mtc_move");
namespace mtc = moveit::task_constructor;

class MTCTaskNode
{
public:
  MTCTaskNode(const rclcpp::NodeOptions& options);

  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr getNodeBaseInterface();

  void doTask();

private:
  // Compose an MTC task from a series of stages.
  mtc::Task createTask();
  mtc::Task task_;
  rclcpp::Node::SharedPtr node_;
};

MTCTaskNode::MTCTaskNode(const rclcpp::NodeOptions& options)
  : node_{ std::make_shared<rclcpp::Node>("cartesian_path_mtc", options) }
{
}

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr MTCTaskNode::getNodeBaseInterface()
{
  return node_->get_node_base_interface();
}

void MTCTaskNode::doTask()
{
  task_ = createTask();

  try
  {
    task_.init();
  }
  catch (mtc::InitStageException& e)
  {
    RCLCPP_ERROR_STREAM(LOGGER, e);
    return;
  }

  if (!task_.plan(5))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
    return;
  }
  task_.introspection().publishSolution(*task_.solutions().front());

  auto result = task_.execute(*task_.solutions().front());
  if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
    return;
  }

  RCLCPP_INFO(LOGGER, "Task execution completed successfully");
  return;
}
mtc::Task MTCTaskNode::createTask()
{
  mtc::Task task;
  task.stages()->setName("UR10 Move to Target Pose");
  task.loadRobotModel(node_);

  const auto& arm_group_name = "ur_arm";
  const auto& base_frame = "ur10_base_link";
  const auto& tcp_frame = "ur10_tcp";

  // Set task properties
  task.setProperty("group", arm_group_name);
  task.setProperty("ik_frame", tcp_frame);

  // Get the current state as the starting point
  auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
  task.add(std::move(stage_state_current));

  // Configure a Cartesian path planner instead of sampling planner
  auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
  cartesian_planner->setMaxVelocityScalingFactor(0.1);  // 30% of maximum velocity
  cartesian_planner->setMaxAccelerationScalingFactor(0.1);  // 20% of maximum acceleration
  cartesian_planner->setStepSize(0.01);  // Small step size for smoother motion
  
  // Define the target pose
  geometry_msgs::msg::PoseStamped target_pose;
  target_pose.header.frame_id = base_frame;
  //target_pose.pose.position.x = 0.734647;
  //target_pose.pose.position.y = 0.303083;
  //target_pose.pose.position.z = -0.436118;
  target_pose.pose.position.x = 0.69327;
  target_pose.pose.position.y = 0.29000;
  target_pose.pose.position.z = -0.435;
  target_pose.pose.orientation.x = -0.070557;
  target_pose.pose.orientation.y = 0.997330;
  target_pose.pose.orientation.z = 0.009466;
  target_pose.pose.orientation.w = -0.016255;

  // Create a move to pose stage with the Cartesian planner
  auto stage_move_to_pose = std::make_unique<mtc::stages::MoveTo>("move to target pose", cartesian_planner);
  stage_move_to_pose->setGroup(arm_group_name);
  stage_move_to_pose->setGoal(target_pose);
  stage_move_to_pose->setIKFrame(tcp_frame);
  
  // Also set the properties at the stage level for redundancy
  //stage_move_to_pose->properties().set("max_velocity_scaling_factor", 0.1);
  //stage_move_to_pose->properties().set("max_acceleration_scaling_factor", 0.1);
  
  task.add(std::move(stage_move_to_pose));

  return task;
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

  mtc_task_node->doTask();

  spin_thread->join();
  rclcpp::shutdown();
  return 0;
}
