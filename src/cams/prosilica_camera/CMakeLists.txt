cmake_minimum_required(VERSION 3.8)
project(prosilica_camera)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_calibration_parsers REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(cam_utils REQUIRED)
find_package(roskv REQUIRED)
find_package(prosilica_gige_sdk REQUIRED)

find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

find_path(HIREDIS_HEADER hiredis)
find_library(HIREDIS_LIB hiredis)
find_path(REDIS_PLUS_PLUS_HEADER sw)
## NOTE: this should be *sw* NOT *redis++*

include_directories(include ${Boost_INCLUDE_DIR})

add_library(prosilica src/libprosilica/prosilica.cpp)
ament_target_dependencies(prosilica rclcpp sensor_msgs prosilica_gige_sdk)
target_link_libraries(prosilica nlohmann_json::nlohmann_json)

add_executable(prosilica_node
        src/nodes/prosilica_node.cpp
        )
ament_target_dependencies(prosilica_node
        rclcpp std_msgs std_srvs sensor_msgs custom_msgs image_transport
        camera_calibration_parsers cv_bridge cam_utils roskv prosilica_gige_sdk)
target_link_libraries(prosilica_node
   prosilica
   ${Boost_LIBRARIES})

add_executable(write_memory src/utilities/write_memory.cpp)
target_link_libraries(write_memory prosilica)

add_executable(read_memory src/utilities/read_memory.cpp)
target_link_libraries(read_memory prosilica
   ${Boost_LIBRARIES})

add_executable(set_ip src/utilities/set_ip.cpp)
target_link_libraries(set_ip prosilica)

add_executable(set_inhibition src/utilities/set_inhibition.cpp)
target_link_libraries(set_inhibition prosilica)

## === === === === === === === Installation

install(TARGETS prosilica_node write_memory read_memory set_ip set_inhibition
   DESTINATION lib/${PROJECT_NAME})

install(TARGETS prosilica
   ARCHIVE DESTINATION lib
   LIBRARY DESTINATION lib)

install(DIRECTORY launch
   DESTINATION share/${PROJECT_NAME}
   FILES_MATCHING PATTERN "*.launch.xml")

ament_package()
