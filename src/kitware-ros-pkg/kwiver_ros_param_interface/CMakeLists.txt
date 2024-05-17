cmake_minimum_required(VERSION 2.8.3)
project(kwiver_ros_param_interface)

add_definitions(-std=c++11)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs)

## Declare a catkin package
#catkin_package()
catkin_package(CATKIN_DEPENDS roscpp std_msgs sensor_msgs kwiver
               INCLUDE_DIRS include)

find_package(KWIVER REQUIRED)
link_directories(${KWIVER_LIBRARY_DIR})
#message("KWIVER_LIBRARIES: ${KWIVER_LIBRARIES}")
include(${KWIVER_CMAKE_DIR}/kwiver-utils.cmake) #kwiver_install_headers, etc

include_directories(include
                    ${catkin_INCLUDE_DIRS}
                    ${KWIVER_INCLUDE_DIRS}
                    ${EIGEN_INCLUDE_DIR}
                    ${CMAKE_CURRENT_BINARY_DIR})

set(dynamic_config_ros_headers_public dynamic_config_ros.h)

## Install KWIVER public header files to include/kwiver/...
kwiver_install_headers(
  SUBDIR     arrows/ros
  ${dynamic_config_ros_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_dynamic_config_ros_export.h
  NOPATH   SUBDIR     arrows/ros
  )

set(dynamic_config_ros_sources dynamic_config_ros.cxx)

## Add a library to Kwiver
kwiver_add_library(kwiver_algo_ros ${dynamic_config_ros_headers_public}
                                   ${dynamic_config_ros_sources})

#target_link_libraries( kwiver_algo_ros
#  PUBLIC               vital_algo
#                       kwiver_algo_core
#  PRIVATE              kwiversys
#                       kwiver_algo_ocv
#  )

#target_link_libraries( kwiver_algo_ros
#   ${catkin_LIBRARIES}
#   ${KWIVER_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )

## Generate and add a plug-in library to Kwiver based on kwiver_algo_ros
#algorithms_create_plugin(kwiver_algo_ros register_algorithms.cxx)

