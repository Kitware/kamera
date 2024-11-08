cmake_minimum_required(VERSION 2.8.3)
project(kw_genicam_driver)
set(CMAKE_CXX_STANDARD 17)
include(CMakePrintHelpers)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  image_transport
        roscpp
        sensor_msgs
        custom_msgs
        camera_info_manager
        cv_bridge
        phase_one
        roskv
        )

find_package(OpenCV  REQUIRED)


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

# GEVLIB
set( GENICAM_LIB GevApi CorW32 )

# specific to the GENICAM SDK installation
# - Binary install root
set( GENICAM_ROOT_V3_0 "/opt/genicam_v3_0" )
# - x86 API (unpacked from tar)
set( DALSA_GIGEV_ROOT "/usr/dalsa/GigeV" )

set( ARCH_GENICAM_BIN Linux64_x64)
set( ARCH_GCCVER 421)

#
# Arch dependent GenICam library specification
#
set( GENICAM_PATH_VERSION v3_0)
set( GENICAM_PATH  ${GENICAM_ROOT_V3_0} )

set( GENICAM_INCLUDE_DIRS
  "${DALSA_GIGEV_ROOT}/include"
  "${GENICAM_PATH}/library/CPP/include"
)
set( GENICAM_LIBRARY_DIRS
  "/usr/local/lib"
  "${GENICAM_PATH}/bin/${ARCH_GENICAM_BIN}"
)
set( GENICAM_LIBRARIES
  ${GENICAM_LIB}
  "GenApi_gcc${ARCH_GCCVER}_${GENICAM_PATH_VERSION}"
  "GCBase_gcc${ARCH_GCCVER}_${GENICAM_PATH_VERSION}"
)

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
#   std_msgs  # Or other packages containing msgs
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
#  LIBRARIES kw_genicam_driver
#  CATKIN_DEPENDS image_transport roscpp sensor_msg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GENICAM_INCLUDE_DIRS}
  ${KWIVER_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}
)

link_directories(${GENICAM_LIBRARY_DIRS}
                 ${KWIVER_LIBRARY_DIR})

## Declare a C++ library
# add_library(kw_genicam_driver
#   src/${PROJECT_NAME}/kw_genicam_driver.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(kw_genicam_driver ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
add_executable( a6750_driver_node
        src/decode_error.cpp
        src/driver_a6750.cpp
        src/macros.h
        src/utils.h
        src/utils.cpp
        src/spec_a6750.h
        src/spec_a6750.cpp
        src/genicam_ctl.cpp src/genicam_ctl.h
        /root/noaa_kamera/src/cams/phase_one/src/phase_one_utils.cpp)

add_executable( gige_scan
        src/gige_scan.cpp
        )


## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(kw_genicam_driver_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries( a6750_driver_node
        PUBLIC ${catkin_LIBRARIES}
        ${GENICAM_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${KWIVER_LIBRARIES}
)

target_link_libraries( gige_scan
        PUBLIC ${catkin_LIBRARIES}
        ${GENICAM_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${KWIVER_LIBRARIES}
        )
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
# install(TARGETS kw_genicam_driver kw_genicam_driver_node
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
# catkin_add_gtest(${PROJECT_NAME}-test test/test_kw_genicam_driver.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
cmake_print_variables(CATKIN_PACKAGE_INCLUDE_DESTINATION catkin_LIBRARIES)
cmake_print_variables(GENICAM_LIBRARIES)
cmake_print_variables(GENICAM_INCLUDE_DIRS)
cmake_print_variables(GENICAM_LIBRARY_DIRS)
