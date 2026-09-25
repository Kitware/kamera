cmake_minimum_required(VERSION 3.8)
project(custom_msgs)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/ArchiveSchema.msg
  msg/Evt.msg
  msg/GsofEvt.msg
  msg/GsofIns.msg
  msg/ImageSpaceDetection.msg
  msg/ImageSpaceDetectionList.msg
  msg/Pashr.msg
  msg/PathImage.msg
  msg/Stat.msg
  msg/SyncedPathImages.msg
  msg/SynchronizedImages.msg
  srv/AddToEventLog.srv
  srv/CamGetAttr.srv
  srv/CamSetAttr.srv
  srv/ReadPin.srv
  srv/RequestCompressedImageView.srv
  srv/RequestImageMetadata.srv
  srv/RequestImageView.srv
  srv/SetArchiving.srv
  srv/SetTriggerRate.srv
  srv/StrList.srv
  srv/SysCall.srv
  srv/TransformDetectionList.srv
  DEPENDENCIES builtin_interfaces std_msgs sensor_msgs geometry_msgs
)

# Python helper library for publishing dispatch wrappers around these messages
ament_python_install_package(msgdispatch PACKAGE_DIR dispatch/msgdispatch)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
