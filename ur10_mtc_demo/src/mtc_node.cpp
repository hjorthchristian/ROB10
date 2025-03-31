#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("ur10_mtc_demo");
namespace mtc = moveit::task_constructor;

class UR10MTCNode
{
public:
  UR10MTCNode(const rclcpp::NodeOptions& options);
  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr getNodeBaseInterface();
  void doTask();

private:
  mtc::Task createTask();
  mtc::Task task_;
  rclcpp::Node::SharedPtr node_;
};

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr UR10MTCNode::getNodeBaseInterface()
{
  return node_->get_node_base_interface();
}

UR10MTCNode::UR10MTCNode(const rclcpp::NodeOptions& options)
  : node_{ std::make_shared<rclcpp::Node>("ur10_mtc_node", options) }
{
}

void UR10MTCNode::doTask()
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
  
  RCLCPP_INFO(LOGGER, "Task completed successfully!");
}

mtc::Task UR10MTCNode::createTask()
{
  mtc::Task task;
  task.stages()->setName("UR10 move down task");
  task.loadRobotModel(node_);
  
  const auto& arm_group_name = "ur_arm";
  const auto& ee_link = "ur10_tcp";
  
  // Set task properties
  task.setProperty("group", arm_group_name);
  task.setProperty("ik_frame", ee_link);
  
  // Get the current state
  auto stage_current_state = std::make_unique<mtc::stages::CurrentState>("current state");
  task.add(std::move(stage_current_state));
  
  // Create planners
  auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
  cartesian_planner->setMaxVelocityScalingFactor(0.10);
  cartesian_planner->setMaxAccelerationScalingFactor(0.10);
  cartesian_planner->setStepSize(0.01);
  
  // Create the move stage
  auto stage_move_down = std::make_unique<mtc::stages::MoveRelative>("move down", cartesian_planner);
  stage_move_down->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
  stage_move_down->setIKFrame(ee_link);
  
  // Set the direction to move 5cm down in Z
  geometry_msgs::msg::Vector3Stamped direction;
  direction.header.frame_id = ee_link;
  direction.vector.z = -0.10;  // 5cm down in Z direction
  
  stage_move_down->setDirection(direction);
  task.add(std::move(stage_move_down));
  RCLCPP_INFO(LOGGER, "Task setup complete");
  
  return task;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  
  auto mtc_node = std::make_shared<UR10MTCNode>(options);
  
  rclcpp::executors::MultiThreadedExecutor executor;
  auto spin_thread = std::make_unique<std::thread>([&executor, &mtc_node]() {
    executor.add_node(mtc_node->getNodeBaseInterface());
    executor.spin();
    executor.remove_node(mtc_node->getNodeBaseInterface());
  });
  
  mtc_node->doTask();
  
  spin_thread->join();
  rclcpp::shutdown();
  return 0;
}
