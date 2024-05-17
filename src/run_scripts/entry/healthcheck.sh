#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
HOST=${REDIS_HOST:-nuvo0}
rosservice call /${HOST}/rgb/rgb_driver/health "{}"
