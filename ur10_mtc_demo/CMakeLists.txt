cmake_minimum_required(VERSION 3.8)
project(ur10_mtc_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pose_estimation_interfaces REQUIRED)
find_package(vg_control_interfaces REQUIRED)
find_package(stack_optimization_interfaces REQUIRED)

add_executable(mtc_node src/mtc_node.cpp)
target_include_directories(mtc_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(mtc_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  mtc_node
  rclcpp
  moveit_task_constructor_core
  tf2_geometry_msgs
  tf2_eigen
)

# Fixed the syntax error - removed extra parenthesis
add_executable(cartesian_path_mtc src/cartesian_path_mtc.cpp)
target_include_directories(cartesian_path_mtc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cartesian_path_mtc PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  cartesian_path_mtc
  rclcpp
  moveit_task_constructor_core
  tf2_geometry_msgs
  tf2_eigen
)

add_executable(pick_and_place_mtc_node src/pick_and_place.cpp)
target_include_directories(pick_and_place_mtc_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(pick_and_place_mtc_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  pick_and_place_mtc_node
  pose_estimation_interfaces
  vg_control_interfaces
  stack_optimization_interfaces
  rclcpp
  moveit_task_constructor_core
  tf2_geometry_msgs
  tf2_eigen
)

add_executable(stacked_box_processor_node src/minimal_task.cpp)
target_include_directories(stacked_box_processor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(stacked_box_processor_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  stacked_box_processor_node
  pose_estimation_interfaces
  vg_control_interfaces
  stack_optimization_interfaces
  rclcpp
  moveit_task_constructor_core
  tf2_geometry_msgs
  tf2_eigen
)


install(TARGETS mtc_node cartesian_path_mtc pick_and_place_mtc_node stacked_box_processor_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()