cmake_minimum_required(VERSION 2.8.3)
project(roskv)
include(CMakePrintHelpers)
set(CMAKE_CXX_STANDARD 17)


## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    rospy
    roscpp
)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_path(HIREDIS_HEADER hiredis)
find_path(REDIS_PLUS_PLUS_HEADER sw)
find_library(HIREDIS_LIB hiredis)
find_library(REDIS_PLUS_PLUS_LIB redis++)

## NOTE: this should be *sw* NOT *redis++*


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS roscpp rospy
 DEPENDS
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

include_directories(include ${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${roscpp_INCLUDE_DIRS})
add_library(roskv
        include/roskv/envoy.h
        include/roskv/archiver.h
        include/roskv/roskv.h
        libroskv/envoy.cpp
        libroskv/archiver.cpp
        libroskv/roskv.cpp
        )
add_executable(test_roskv libroskv/test_roskv.cpp)
target_link_libraries(roskv
        fmt::fmt
        ${catkin_LIBRARIES}
        ${nlohmann_json_LIBRARIES}
        ${HIREDIS_LIB}
        ${REDIS_PLUS_PLUS_LIB})

target_link_libraries(test_roskv
        fmt::fmt
        ${catkin_LIBRARIES}
        ${nlohmann_json_LIBRARIES}
        ${HIREDIS_LIB}
        ${REDIS_PLUS_PLUS_LIB})
target_include_directories(test_roskv PUBLIC ${REDIS_PLUS_PLUS_HEADER})


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
install(TARGETS roskv
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )
install(
        DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

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
# catkin_add_gtest(${PROJECT_NAME}-test test/test_nexus.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
cmake_print_variables(PROJECT_NAME)
cmake_print_variables(CATKIN_PACKAGE_INCLUDE_DESTINATION catkin_LIBRARIES)
cmake_print_variables(fmt_FOUND FMT_HEADER FMT_LIB fmt_LIBRARIES)
cmake_print_variables(nlohmann_json_LIBRARIES REDIS_PLUS_PLUS_HEADER REDIS_PLUS_PLUS_LIB)
