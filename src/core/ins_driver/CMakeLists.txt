cmake_minimum_required(VERSION 2.8.3)
project(ins_driver)

find_package(catkin REQUIRED)

catkin_python_setup()
catkin_package()

install(PROGRAMS
   scripts/nmea_serial_driver
   scripts/nmea_socket_driver
   scripts/nmea_topic_driver
   scripts/nmea_topic_serial_reader
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#if (CATKIN_ENABLE_TESTING)
#  find_package(roslint)
#  roslint_python()
#  roslint_add_test()
#endif()
