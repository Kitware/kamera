cmake_minimum_required(VERSION 3.8)
project(bag_file_explode)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/rosbag_to_png.py
  scripts/explode_bag_files.sh
  scripts/explode_directories.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
