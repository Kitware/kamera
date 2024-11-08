cmake_minimum_required(VERSION 3.1.0)
project(sprokit_adapters)
set(CMAKE_CXX_STANDARD 17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        roscpp
        std_msgs
        custom_msgs
        image_transport
        eigen_conversions
        roskv
        )

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
#find_package(fletch REQUIRED)
#find_package(kwiver REQUIRED)
#find_package(OpenCV  REQUIRED)
catkin_package()

###########
## Build ##
###########
find_package(KWIVER REQUIRED)

set(KWIVER_CMAKE_DIR "/opt/noaa/viame/lib/cmake/kwiver/")
set(KWIVER_INCLUDE_DIR "/opt/noaa/viame/include/")
set(KWIVER_LIBRARY_DIR "/opt/noaa/viame/lib")
set(KWIVER_INCLUDE_DIRS "/opt/noaa/viame/include;/opt/noaa/viame/include/eigen3;/opt/noaa/viame/include")
message("KWIVER_INCLUDE_DIR=${KWIVER_INCLUDE_DIR}")
list( INSERT CMAKE_MODULE_PATH 0 "${KWIVER_CMAKE_DIR}" )

include(kwiver-cmake-future)
include(kwiver-utils)

add_definitions(-std=c++17)


## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${KWIVER_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIR}
        ${cv_bridge_INCLUDE_DIRS}
        ${CMAKE_BINARY_DIR}
)

## Declare a C++ executable
link_directories( ${KWIVER_LIBRARY_DIR} )

add_executable(kw_detector_fusion_adapter_node src/kw_detector_fusion_adapter.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(sprokit_adapters_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# <------------ add hiredis dependency --------------->
find_path(HIREDIS_HEADER hiredis)
find_library(HIREDIS_LIB hiredis)
# <------------ add redis-plus-plus dependency -------------->
# NOTE: this should be *sw* NOT *redis++*
find_path(REDIS_PLUS_PLUS_HEADER sw)
find_library(REDIS_PLUS_PLUS_LIB redis++)

## Specify libraries to link a library or executable target against
target_link_libraries( kw_detector_fusion_adapter_node
        ${catkin_LIBRARIES}
        ${KWIVER_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${HIREDIS_LIB}
        ${REDIS_PLUS_PLUS_LIB}
        sprokit_pipeline sprokit_pipeline_util
        )

###
# Add detector scaling plugin
kwiver_add_plugin( ros_detector_scaling
        SOURCES         src/ros_detector_scaling.h
        src/ros_detector_scaling.cxx
        PRIVATE         vital
        vital_algo
        vital_vpm
        vital_config
        ${catkin_LIBRARIES}
        )

# need to install plugin in kwiver area to simplify loading.
set_target_properties( ros_detector_scaling
        PROPERTIES  LIBRARY_OUTPUT_DIRECTORY  ${KWIVER_MODULE_DIR} )

###
# Add dynamic config provider
kwiver_add_plugin( ros_dynamic_config
        SOURCES         src/ros_dynamic_config.h
        src/ros_dynamic_config.cxx
        PRIVATE         vital
        vital_algo
        vital_vpm
        vital_config
        ${catkin_LIBRARIES}
        )

# need to install plugin in kwiver area to simplify loading.
set_target_properties( ros_dynamic_config
        PROPERTIES  LIBRARY_OUTPUT_DIRECTORY  ${KWIVER_MODULE_DIR} )
