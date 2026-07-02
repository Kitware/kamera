cmake_minimum_required(VERSION 3.8)
project(mcc_daq)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(roskv REQUIRED)

find_library(LIBUSB NAMES usb-1.0)
find_path(LIBUSB_INCLUDE_DIR NAMES libusb.h PATH_SUFFIXES "include" "libusb" "libusb-1.0")

find_library(LIBHIDAPI NAMES hidapi hidapi-libusb)
find_path(LIBHIDAPI_INCLUDE_DIR hidapi.h PATH_SUFFIXES "hidapi")

message(STATUS "FINDING USB and HID: ${LIBUSB_INCLUDE_DIR} ${LIBUSB} ${LIBHIDAPI_INCLUDE_DIR} ${LIBHIDAPI}")

include_directories(/usr/include/libusb-1.0/)

add_executable(ctest-usb2408
        src/nist.c
        src/pmd.c
        src/usb-2408.c
        src/usb-2416.c
        src/test-usb2408.c
        )
target_link_libraries(ctest-usb2408 ${LIBUSB} ${LIBHIDAPI} m)

add_executable(daq_node
        src/nist.c
        src/pmd.c
        src/usb-2408.c
        src/usb-2416.c
        src/daq_node.cpp
        src/usbdaq.cpp
        src/utils.cpp)
ament_target_dependencies(daq_node rclcpp std_msgs custom_msgs roskv)
target_link_libraries(daq_node
        ${LIBUSB}
        ${LIBHIDAPI}
        )

install(TARGETS daq_node ctest-usb2408
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME})

ament_package()
