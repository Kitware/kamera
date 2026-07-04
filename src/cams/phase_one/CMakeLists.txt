cmake_minimum_required(VERSION 3.13)
project(phase_one)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME_LC)
set(SDK_PACKAGE_EXT ".tgz")
set(SDK_MAJOR_VERSION "3")
set(SDK_MINOR_VERSION "1")

if(NOT APPLE AND UNIX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(LINUX_ARCH "-arm64")
else()
    set(LINUX_ARCH "")
endif()

# refs
# https://developer.phaseone.com/sdk/3.1/releases/imagesdk/3/p1imagesdk-linux.tgz
# https://developer.phaseone.com/sdk/3.1/releases/camerasdk/3/p1camerasdk-linux.tgz

FetchContent_Declare(CameraSDK
    URL https://developer.phaseone.com/sdk/${SDK_MAJOR_VERSION}.${SDK_MINOR_VERSION}/releases/camerasdk/${SDK_MAJOR_VERSION}/p1camerasdk-${SYSTEM_NAME_LC}${LINUX_ARCH}${SDK_PACKAGE_EXT}
    SOURCE_DIR CameraSDK
)

message(STATUS "Downloading CameraSDK...")
FetchContent_Populate(CameraSDK)

find_package(CameraSDK CONFIG REQUIRED HINTS ${CMAKE_CURRENT_BINARY_DIR}/CameraSDK)

# Leave it to a copy for the GPU version, rather than fetching
find_package(ImageSDK CONFIG REQUIRED HINTS lib/ImageSDKCuda)

find_package(OpenCV REQUIRED
    COMPONENTS
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
    CONFIG
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(cam_utils REQUIRED)
find_package(roskv REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(Boost REQUIRED COMPONENTS system)

find_package(CUDA QUIET)
if(NOT CUDA_FOUND)
    message(WARNING "CUDA not found - nvjpeg will not be available")
endif()

## Find nvjpeg (part of CUDA Toolkit)
if(CUDA_FOUND)
    find_library(NVJPEG_LIBRARY
        NAMES nvjpeg
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
    )
    if(NVJPEG_LIBRARY)
        set(NVJPEG_FOUND TRUE)
        message(STATUS "nvjpeg library found: ${NVJPEG_LIBRARY}")
    else()
        set(NVJPEG_FOUND FALSE)
        message(WARNING "nvjpeg library not found - nvjpeg encoding will not be available")
    endif()

    # Find nvjpeg header
    find_path(NVJPEG_INCLUDE_DIR
        NAMES nvjpeg.h
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include
    )
    if(NVJPEG_INCLUDE_DIR)
        message(STATUS "nvjpeg headers found: ${NVJPEG_INCLUDE_DIR}")
        add_definitions(-DHAVE_NVJPEG)
    else()
        message(WARNING "nvjpeg headers not found")
    endif()
endif()

## Services
rosidl_generate_interfaces(${PROJECT_NAME}
    srv/GetPhaseOneParameter.srv
    srv/SetPhaseOneParameter.srv
    srv/GetCompressedImageView.srv
    srv/GetImageView.srv
    DEPENDENCIES std_msgs sensor_msgs
)

include_directories(include ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
if(NVJPEG_FOUND AND NVJPEG_INCLUDE_DIR)
    include_directories(${NVJPEG_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
endif()

add_executable(${PROJECT_NAME}_standalone src/phase_one_standalone.cpp)
ament_target_dependencies(${PROJECT_NAME}_standalone
    rclcpp std_msgs sensor_msgs custom_msgs image_transport cv_bridge cam_utils roskv)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(${PROJECT_NAME}_standalone
    "${cpp_typesupport_target}"
    ${OpenCV_LIBS}
    CameraSDK::CameraSdkCpp
    ImageSDK::ImageSdkCpp
)
if(NVJPEG_FOUND)
    target_link_libraries(${PROJECT_NAME}_standalone
        ${NVJPEG_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
    )
endif()

install(TARGETS ${PROJECT_NAME}_standalone
    DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.launch.xml")
install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
