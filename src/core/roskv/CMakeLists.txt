cmake_minimum_required(VERSION 3.8)
project(roskv)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(nlohmann_json REQUIRED)
find_path(HIREDIS_HEADER hiredis)
find_path(REDIS_PLUS_PLUS_HEADER sw)
find_library(HIREDIS_LIB hiredis)
find_library(REDIS_PLUS_PLUS_LIB redis++)

add_library(roskv
  libroskv/envoy.cpp
  libroskv/archiver.cpp
)
target_include_directories(roskv PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${REDIS_PLUS_PLUS_HEADER}
)
target_link_libraries(roskv
  nlohmann_json::nlohmann_json
  ${HIREDIS_LIB}
  ${REDIS_PLUS_PLUS_LIB}
)

add_executable(test_roskv libroskv/test_roskv.cpp)
target_link_libraries(test_roskv roskv)

install(TARGETS roskv
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# Pure-python KV library + CLI
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/roskv)
install(PROGRAMS scripts/roskv DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(nlohmann_json)
ament_package()
