cmake_minimum_required(VERSION 0.0.1)
project(ser_daq)

find_package(catkin REQUIRED)

catkin_python_setup()
catkin_package()

install(PROGRAMS
   scripts/ser_daq_driver
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#if (CATKIN_ENABLE_TESTING)
#  find_package(roslint)
#  roslint_python()
#  roslint_add_test()
#endif()
