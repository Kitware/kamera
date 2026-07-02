cmake_minimum_required(VERSION 3.8)
project(kw_genicam_driver)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CMakePrintHelpers)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(cam_utils REQUIRED)
find_package(roskv REQUIRED)
find_package(OpenCV REQUIRED)

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

include_directories(${GENICAM_INCLUDE_DIRS})
link_directories(${GENICAM_LIBRARY_DIRS})

add_executable( a6750_driver_node
        src/decode_error.cpp
        src/driver_a6750.cpp
        src/utils.cpp
        src/spec_a6750.cpp
        src/genicam_ctl.cpp)
ament_target_dependencies(a6750_driver_node
        rclcpp std_msgs sensor_msgs custom_msgs image_transport cv_bridge cam_utils roskv)
target_link_libraries( a6750_driver_node
        ${GENICAM_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

add_executable( gige_scan
        src/gige_scan.cpp
        )
target_link_libraries( gige_scan
        ${GENICAM_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

install(TARGETS a6750_driver_node gige_scan
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.launch.xml")

ament_package()
