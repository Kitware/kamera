cmake_minimum_required(VERSION 2.8.3)
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

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES sprokit_adapters
        #  CATKIN_DEPENDS roscpp std_msgs
        #  DEPENDS system_lib
)

###########
## Build ##
###########
find_package(KWIVER REQUIRED)

include(kwiver-utils)

add_definitions(-std=c++11)

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

## Declare a C++ library
# add_library(sprokit_adapters
#   src/${PROJECT_NAME}/sprokit_adapters.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(sprokit_adapters ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
link_directories(${KWIVER_LIBRARY_DIR})
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

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS sprokit_adapters sprokit_adapters_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_sprokit_adapters.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
