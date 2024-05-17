#!/bin/bash

echo "<> <> <> ROS2JAEGER <> <> <> "
source /entry/project.sh

ROSWAIT="--wait"

catkin build ros2jaeger

exec roslaunch "${ROSWAIT}" ros2jaeger ros2jaeger.launch \
    system_name:=${NODE_HOSTNAME} \
