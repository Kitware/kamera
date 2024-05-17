cmake_minimum_required(VERSION 2.8.3)
project(prosilica_camera)
set(CMAKE_CXX_STANDARD 17)

# Load catkin and all dependencies required for this package
# TODO: remove all from COMPONENTS that are not catkin packages.

find_package(catkin REQUIRED COMPONENTS
   prosilica_gige_sdk
   roscpp
   std_msgs
   std_srvs
   message_generation
   sensor_msgs
   custom_msgs
   diagnostic_updater
   image_transport
   self_test
   rosconsole
   phase_one
   dynamic_reconfigure
   camera_calibration_parsers
   polled_camera
   nodelet
   nodelet_topic_tools
   cv_bridge
   roskv
   )

find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)

find_package(Boost REQUIRED COMPONENTS thread)

# <------------ add hiredis dependency --------------->
find_path(HIREDIS_HEADER hiredis)
find_library(HIREDIS_LIB hiredis)
find_path(REDIS_PLUS_PLUS_HEADER sw)
## NOTE: this should be *sw* NOT *redis++*

include_directories(include ${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${roscpp_INCLUDE_DIRS})

# Generate dynamic parameters
generate_dynamic_reconfigure_options(cfg/ProsilicaCamera.cfg)

# Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
   sensor_msgs
 )

catkin_package()

string (REPLACE ";" " " catkin_INCLUDE_STR "${catkin_INCLUDE_DIRS}")

message(STATUS "Catkin include: ${catkin_INCLUDE_DIRS}")
message(STATUS "Catkin include: ${catkin_INCLUDE_STR}")

add_library(prosilica src/libprosilica/prosilica.cpp)
target_link_libraries(prosilica ${catkin_LIBRARIES})

add_library(prosilica_nodelet
        src/nodes/prosilica_nodelet.cpp
        src/nodes/view_server_nodelet.cpp
        /root/kamera_ws/src/cams/phase_one/src/phase_one_utils.cpp
        )
target_link_libraries(prosilica_nodelet
        prosilica
        ${nlohman_json_LIBRARIES}
        ${catkin_LIBRARIES}
        fmt::fmt
        )
target_include_directories(prosilica_nodelet PUBLIC /opt/ros/noetic/include)
class_loader_hide_library_symbols(prosilica_nodelet)

add_library(view_server_nodelet
         src/nodes/view_server_nodelet.cpp
         )

add_executable(prosilica_node
        src/nodes/prosilica_node.cpp
        )
target_link_libraries(prosilica_node
   prosilica
   prosilica_nodelet
   view_server_nodelet
   ${Boost_LIBRARIES}
   ${catkin_LIBRARIES})
target_include_directories(prosilica_node PUBLIC /opt/ros/noetic/include)
add_dependencies(
    prosilica_camera_gencpp
    prosilica_node
   ${catkin_EXPORTED_TARGETS}
   ${prosilica_camera_EXPORTED_TARGETS})

# disabled until it's mature??
#rosbuild_add_executable(find_camera find_camera.cpp)

add_executable(write_memory src/utilities/write_memory.cpp)
target_link_libraries(write_memory prosilica ${catkin_LIBRARIES})

add_executable(read_memory src/utilities/read_memory.cpp)
target_link_libraries(read_memory prosilica
   ${catkin_LIBRARIES}
   ${Boost_LIBRARIES})

add_executable(set_ip src/utilities/set_ip.cpp)
target_link_libraries(set_ip prosilica ${catkin_LIBRARIES})

add_executable(set_inhibition src/utilities/set_inhibition.cpp)
target_link_libraries(set_inhibition prosilica ${catkin_LIBRARIES})

## === === === === === === === Installation

install(TARGETS prosilica_node write_memory read_memory set_ip set_inhibition
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(TARGETS prosilica prosilica_nodelet
   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(DIRECTORY launch
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(FILES prosilica.launch streaming.launch plugins/nodelet_plugins.xml
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


# Clion hackery
message(STATUS "Cmake include: ${CMAKE_INCLUDE_PATH}")
