cmake_minimum_required(VERSION 3.0.2)
project(mpc)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  nav_msgs
  std_msgs
  # message_generation
)

find_package(Eigen3 REQUIRED)
# find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(PCL 1.10 REQUIRED)

add_message_files(FILES
  Polynome.msg
)

generate_messages(DEPENDENCIES
	std_msgs
  geometry_msgs
)

catkin_package(
 CATKIN_DEPENDS roscpp std_msgs
 message_runtime
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

add_library(mpc
  src/mpc.cpp
  src/cubic_spline_planner.cpp
)

add_dependencies(mpc mpc_gencpp)

target_link_libraries(mpc
    OsqpEigen::OsqpEigen
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
    )

add_executable(mpc_controller_node
    src/mpc_node.cpp
  )
target_link_libraries(mpc_controller_node
  mpc
  ${PCL_LIBRARIES}
  ${catkin_LIBRARIES}
)