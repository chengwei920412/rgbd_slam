cmake_minimum_required(VERSION 2.8.3)
project(rgbd_slam)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  rospy
  sensor_msgs
  std_msgs
  pcl_conversions
  pcl_ros roscpp
  sensor_msgs
)


catkin_package(
)

###########
## Build ##
###########

include_directories(
  include ${catkin_INCLUDE_DIRS}
)




## Declare a C++ executable
add_executable(generatePointCloud src/generatePointCloud.cpp src/slamBase.cpp)
add_executable(detect_features src/detect_features.cpp)


## Specify libraries to link a library or executable target against
target_link_libraries(detect_features ${catkin_LIBRARIES})
target_link_libraries(generatePointCloud ${catkin_LIBRARIES})
