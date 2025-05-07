cmake_minimum_required(VERSION 3.8)
project(moveit_ur)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(pose_estimation_interfaces REQUIRED)
find_package(vg_control_interfaces REQUIRED)
find_package(stack_optimization_interfaces REQUIRED)

add_executable(moveit_ur src/moveit_ur.cpp)
target_include_directories(moveit_ur PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(moveit_ur PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  moveit_ur
  "moveit_ros_planning_interface"
  "rclcpp"
  "moveit_visual_tools"

)
add_executable(move_group_interface_ur src/moveit_group_interface_ur.cpp)
target_include_directories(move_group_interface_ur PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(move_group_interface_ur PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  move_group_interface_ur
  "moveit_ros_planning_interface"
  "rclcpp"
  "moveit_visual_tools"
)

add_executable(get_tcp_pose src/get_tcp_pose.cpp)
target_include_directories(get_tcp_pose PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(get_tcp_pose PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(get_tcp_pose
  rclcpp
  moveit_ros_planning_interface
  # Add other dependencies if required
)

# Add the new move_to_pose_move_group executable
add_executable(move_to_pose_movegroup src/move_to_pose_movegroup.cpp)
target_include_directories(move_to_pose_movegroup PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(move_to_pose_movegroup PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(move_to_pose_movegroup
  rclcpp
  moveit_ros_planning_interface
  pose_estimation_interfaces
  vg_control_interfaces
  stack_optimization_interfaces

)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS moveit_ur move_group_interface_ur get_tcp_pose move_to_pose_movegroup
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
