cmake_minimum_required(VERSION 3.8)
project(sprokit_adapters)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(roskv REQUIRED)

find_package(KWIVER REQUIRED)

set(KWIVER_CMAKE_DIR "/opt/noaa/viame/lib/cmake/kwiver/")
set(KWIVER_INCLUDE_DIR "/opt/noaa/viame/include/")
set(KWIVER_LIBRARY_DIR "/opt/noaa/viame/lib")
set(KWIVER_INCLUDE_DIRS "/opt/noaa/viame/include;/opt/noaa/viame/include/eigen3;/opt/noaa/viame/include")
message("KWIVER_INCLUDE_DIR=${KWIVER_INCLUDE_DIR}")
list( INSERT CMAKE_MODULE_PATH 0 "${KWIVER_CMAKE_DIR}" )

include(kwiver-cmake-future)
include(kwiver-utils)

include_directories(
        ${KWIVER_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_BINARY_DIR}
)

link_directories( ${KWIVER_LIBRARY_DIR} )

add_executable(kw_detector_fusion_adapter_node src/kw_detector_fusion_adapter.cpp)
ament_target_dependencies(kw_detector_fusion_adapter_node
        rclcpp std_msgs sensor_msgs custom_msgs cv_bridge roskv)
target_link_libraries( kw_detector_fusion_adapter_node
        kwiver_adapter
        vital
        vital_config
        vital_exceptions
        vital_logger
        vital_util
        kwiver_algo_ocv
        sprokit_pipeline
        sprokit_pipeline_util
        ${OpenCV_LIBRARIES}
        )

install(TARGETS kw_detector_fusion_adapter_node
        DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
        scripts/publish_sync_msgs.py
        scripts/rebroadcast_infrequent_detections.py
        scripts/save_detection_chips_to_disk.py
        scripts/save_images_to_disk.py
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.launch.xml")
install(DIRECTORY pipelines
        DESTINATION share/${PROJECT_NAME})

ament_package()
