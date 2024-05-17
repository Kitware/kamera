cmake_minimum_required(VERSION 2.8.3)
project(kamcore)

find_package(catkin REQUIRED)

catkin_python_setup()
catkin_package()

install(PROGRAMS
   scripts/kamcore_node.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#if (CATKIN_ENABLE_TESTING)
#  find_package(roslint)
#  roslint_python()
#  roslint_add_test()
#endif()
