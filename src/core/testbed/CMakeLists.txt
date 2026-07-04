cmake_minimum_required(VERSION 3.8)
project(testbed)
include(CMakePrintHelpers)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(roskv REQUIRED)
find_package(nlohmann_json REQUIRED)
find_path(HIREDIS_HEADER hiredis)
find_path(REDIS_PLUS_PLUS_HEADER sw)
find_library(HIREDIS_LIB hiredis)
find_library(REDIS_PLUS_PLUS_LIB redis++)

add_executable(test_roskv_node src/test_roskv.cpp)
ament_target_dependencies(test_roskv_node rclcpp roskv)
target_link_libraries(test_roskv_node
        nlohmann_json::nlohmann_json
        ${HIREDIS_LIB}
        ${REDIS_PLUS_PLUS_LIB})
target_include_directories(test_roskv_node PUBLIC ${REDIS_PLUS_PLUS_HEADER})

install(TARGETS test_roskv_node DESTINATION lib/${PROJECT_NAME})

ament_package()
