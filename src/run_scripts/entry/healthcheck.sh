#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
HOST=${REDIS_HOST:-nuvo0}
ros2 service call /${HOST}/rgb/rgb_driver/health std_srvs/srv/Trigger "{}"
