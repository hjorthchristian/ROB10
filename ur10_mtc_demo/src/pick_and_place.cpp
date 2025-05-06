#include <rclcpp/rclcpp.hpp>

// MoveIt Task Constructor
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#include <moveit_task_constructor_msgs/action/execute_task_solution.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>

// ROS2 message types
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>

// Custom perception & stacking services
#include "pose_estimation_interfaces/srv/pose_estimation.hpp"
#include "stack_optimization_interfaces/srv/stack_optimizer.hpp"

// Vacuum gripper services
#include "vg_control_interfaces/srv/vacuum_set.hpp"
#include "vg_control_interfaces/srv/vacuum_release.hpp"

using PoseEstimationSrv = pose_estimation_interfaces::srv::PoseEstimation;
using StackOptimizerSrv = stack_optimization_interfaces::srv::StackOptimizer;
using VacuumSetSrv = vg_control_interfaces::srv::VacuumSet;
using VacuumReleaseSrv = vg_control_interfaces::srv::VacuumRelease;

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("pick_and_place_mtc_node");

    // Service clients
    auto pose_client = node->create_client<PoseEstimationSrv>("pose_estimation");
    auto stack_client = node->create_client<StackOptimizerSrv>("stack_optimizer");
    auto grip_client = node->create_client<VacuumSetSrv>("grip_adjust");
    auto release_client = node->create_client<VacuumReleaseSrv>("release_vacuum");

    pose_client->wait_for_service();
    stack_client->wait_for_service();
    grip_client->wait_for_service();
    release_client->wait_for_service();

    // MoveIt group & frame names
    const std::string ARM_GROUP = "ur_arm";
    const std::string EE_LINK = "ur10_tcp";
    const std::string IK_FRAME = "ur10_tcp";
    const std::string BASE_FRAME = "ur10_base_link";

    // Planners for MTC
    auto cartesian_planner = std::make_shared<moveit::task_constructor::solvers::CartesianPath>();
    auto pipeline_planner = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node);
    auto interpolation_planner = std::make_shared<moveit::task_constructor::solvers::JointInterpolationPlanner>();

    // Spinner
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinner([&executor]() { executor.spin(); });

    while (rclcpp::ok())
    {
        // 1) Detect box
        auto pose_req = std::make_shared<PoseEstimationSrv::Request>();
        auto pose_fut = pose_client->async_send_request(pose_req);

        if (rclcpp::spin_until_future_complete(node, pose_fut)
            != rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_ERROR(node->get_logger(), "PoseEstimation failed");
            break;
        }

        auto pose_res = pose_fut.get();
        if (!pose_res->success || pose_res->orientations.empty())
        {
            RCLCPP_INFO(node->get_logger(), "No more boxes.");
            break;
        }

        // Extract box data
        auto box_pos = pose_res->position;
        auto box_orient = pose_res->orientations[0];
        
        // Note: Use actual field names from your service definition
        // These are placeholder names based on your errors
        double box_h = pose_res->z_height * 0.001; // mm → m

        RCLCPP_INFO(node->get_logger(),
            "Box at [%.3f, %.3f, %.3f], height=%.3f m",
            box_pos.x, box_pos.y, box_pos.z, box_h);

        // 2) Compute placement
        auto stack_req = std::make_shared<StackOptimizerSrv::Request>();
        
        // Note: Use actual field names from your service definition
        stack_req->width = static_cast<uint16_t>(pose_res->x_width);
        stack_req->length = static_cast<uint16_t>(pose_res->y_length);
        stack_req->height = static_cast<uint16_t>(pose_res->z_height);
        stack_req->orientation = box_orient;
        stack_req->change_stack_allowed = false;

        auto stack_fut = stack_client->async_send_request(stack_req);

        if (rclcpp::spin_until_future_complete(node, stack_fut)
            != rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_ERROR(node->get_logger(), "StackOptimizer failed");
            break;
        }

        auto stack_res = stack_fut.get();
        if (!stack_res->success)
        {
            RCLCPP_ERROR(node->get_logger(), "StackOptimizer returned error");
            continue;
        }

        auto place_pos = stack_res->position;
        auto place_orient = stack_res->orientations.empty()
            ? box_orient
            : stack_res->orientations[0];

        RCLCPP_INFO(node->get_logger(),
            "Place target at [%.3f, %.3f, %.3f]",
            place_pos.x, place_pos.y, place_pos.z);

        // 3) Build and run MTC task
        moveit::task_constructor::Task task;
        task.loadRobotModel(node);
        task.setProperty("group", ARM_GROUP);
        task.setProperty("ik_frame", IK_FRAME);

        // -- current state --
        task.add(std::make_unique<
            moveit::task_constructor::stages::CurrentState>("current state"));

        // -- open gripper --
        auto open_gripper = std::make_unique<
            moveit::task_constructor::stages::ModifyPlanningScene>("open gripper");

        open_gripper->setCallback(
            [&](const planning_scene::PlanningScenePtr& scene, const moveit::task_constructor::PropertyMap& properties) {
                auto req = std::make_shared<VacuumSetSrv::Request>();
                req->channel_a = 0;
                req->channel_b = 0;
                auto f = grip_client->async_send_request(req);
                if (rclcpp::spin_until_future_complete(node, f)
                    != rclcpp::FutureReturnCode::SUCCESS
                    || !f.get()->success)
                    throw moveit::task_constructor::InitStageException(
                        *open_gripper, "Failed to open gripper");
            });

        task.add(std::move(open_gripper));

        // -- move to pre-grasp --
        geometry_msgs::msg::PoseStamped pre_grasp;
        pre_grasp.header.frame_id = BASE_FRAME;
        pre_grasp.pose.position = box_pos;
        pre_grasp.pose.orientation = box_orient;
        pre_grasp.pose.position.z += (box_h/2 + 0.10);

        auto m = std::make_unique<
            moveit::task_constructor::stages::MoveTo>(
            "move to pre-grasp", pipeline_planner);
        m->setGroup(ARM_GROUP);
        m->setIKFrame(IK_FRAME);
        m->setGoal(pre_grasp);
        task.add(std::move(m));

        // -- approach down --
        auto a = std::make_unique<
            moveit::task_constructor::stages::MoveRelative>(
            "approach", cartesian_planner);
        a->setGroup(ARM_GROUP);
        a->setIKFrame(IK_FRAME);
        
        geometry_msgs::msg::Vector3Stamped approach_dir;
        approach_dir.header.frame_id = IK_FRAME;
        approach_dir.vector.z = -1.0;
        a->setDirection(approach_dir);
        a->setMinMaxDistance(0.05, 0.10);
        task.add(std::move(a));

        // -- attach box --
        auto att = std::make_unique<
            moveit::task_constructor::stages::ModifyPlanningScene>(
            "attach object");
        att->attachObject("box", EE_LINK);
        task.add(std::move(att));

        // -- close gripper --
        auto close_gripper = std::make_unique<
            moveit::task_constructor::stages::ModifyPlanningScene>("close gripper");

        close_gripper->setCallback(
            [&](const planning_scene::PlanningScenePtr& scene, const moveit::task_constructor::PropertyMap& properties) {
                auto req = std::make_shared<VacuumSetSrv::Request>();
                req->channel_a = 150;
                req->channel_b = 150;
                auto f = grip_client->async_send_request(req);
                if (rclcpp::spin_until_future_complete(node, f)
                    != rclcpp::FutureReturnCode::SUCCESS
                    || !f.get()->success)
                    throw moveit::task_constructor::InitStageException(
                        *close_gripper, "Failed to close gripper");
            });

        task.add(std::move(close_gripper));

        // -- lift up --
        auto lift = std::make_unique<
            moveit::task_constructor::stages::MoveRelative>(
            "lift", cartesian_planner);
        lift->setGroup(ARM_GROUP);
        lift->setIKFrame(IK_FRAME);
        
        geometry_msgs::msg::Vector3Stamped lift_dir;
        lift_dir.header.frame_id = BASE_FRAME;
        lift_dir.vector.z = 1.0;
        lift->setDirection(lift_dir);
        lift->setMinMaxDistance(0.05, 0.15);
        task.add(std::move(lift));

        // -- **new safety waypoint** --
        geometry_msgs::msg::PoseStamped waypoint;
        waypoint.header.frame_id = BASE_FRAME;
        waypoint.pose.position.x = 0.823;
        waypoint.pose.position.y = 0.824;
        // sit at top of box + 0.30 m
        waypoint.pose.position.z = box_pos.z + (box_h/2 + 0.30);
        waypoint.pose.orientation = box_orient;

        auto wp = std::make_unique<
            moveit::task_constructor::stages::MoveTo>(
            "move to waypoint", pipeline_planner);
        wp->setGroup(ARM_GROUP);
        wp->setIKFrame(IK_FRAME);
        wp->setGoal(waypoint);
        task.add(std::move(wp));

        // -- move to pre-place --
        geometry_msgs::msg::PoseStamped pre_place;
        pre_place.header.frame_id = BASE_FRAME;
        pre_place.pose.position = place_pos;
        pre_place.pose.orientation = place_orient;
        pre_place.pose.position.z += (box_h/2 + 0.10);

        auto m2 = std::make_unique<
            moveit::task_constructor::stages::MoveTo>(
            "move to pre-place", pipeline_planner);
        m2->setGroup(ARM_GROUP);
        m2->setIKFrame(IK_FRAME);
        m2->setGoal(pre_place);
        task.add(std::move(m2));

        // -- lower, release, detach, retreat --
        auto lower = std::make_unique<
            moveit::task_constructor::stages::MoveRelative>(
            "lower", cartesian_planner);
        lower->setGroup(ARM_GROUP);
        lower->setIKFrame(IK_FRAME);
        
        geometry_msgs::msg::Vector3Stamped lower_dir;
        lower_dir.header.frame_id = BASE_FRAME;
        lower_dir.vector.z = -1.0;
        lower->setDirection(lower_dir);
        lower->setMinMaxDistance(0.05, 0.15);
        task.add(std::move(lower));

        auto release_gripper = std::make_unique<
            moveit::task_constructor::stages::ModifyPlanningScene>("release gripper");

        release_gripper->setCallback(
            [&](const planning_scene::PlanningScenePtr& scene, const moveit::task_constructor::PropertyMap& properties) {
                auto req = std::make_shared<VacuumReleaseSrv::Request>();
                req->release_vacuum = 1;
                auto f = release_client->async_send_request(req);
                if (rclcpp::spin_until_future_complete(node, f)
                    != rclcpp::FutureReturnCode::SUCCESS
                    || !f.get()->success)
                    throw moveit::task_constructor::InitStageException(
                        *release_gripper, "Failed to release gripper");
            });

        task.add(std::move(release_gripper));

        auto det = std::make_unique<
            moveit::task_constructor::stages::ModifyPlanningScene>(
            "detach object");
        det->detachObject("box", EE_LINK);
        task.add(std::move(det));

        auto retreat = std::make_unique<
            moveit::task_constructor::stages::MoveRelative>(
            "retreat", cartesian_planner);
        retreat->setGroup(ARM_GROUP);
        retreat->setIKFrame(IK_FRAME);
        
        geometry_msgs::msg::Vector3Stamped retreat_dir;
        retreat_dir.header.frame_id = IK_FRAME;
        retreat_dir.vector.z = 1.0;
        retreat->setDirection(retreat_dir);
        retreat->setMinMaxDistance(0.05, 0.15);
        task.add(std::move(retreat));

        // Plan & execute
        try {
            task.init();
        } catch (const moveit::task_constructor::InitStageException& e) {
            RCLCPP_ERROR(node->get_logger(), "MTC init error: %s", e.what());
            break;
        }

        if (!task.plan(10)) {
            RCLCPP_ERROR(node->get_logger(), "Task planning failed");
            continue;
        }

        RCLCPP_INFO(node->get_logger(), "Executing plan...");
        const auto& sol = task.solutions().front();
        task.introspection().publishSolution(*sol);
        auto result = task.execute(*sol);
        if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
            RCLCPP_ERROR(node->get_logger(), "Execution failed: %d", result.val);
        else
            RCLCPP_INFO(node->get_logger(), "Pick-and-place succeeded");
    }

    rclcpp::shutdown();
    spinner.join();
    return 0;
}
