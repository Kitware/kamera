#!/bin/bash

# Core bootstrap script.
# In ROS1 this container ran roscore + a keepalive node and loaded the global
# rosparam tree. ROS2 has no master and the config lives in Redis, so all
# that's left is seeding Redis with the static system config.

echo "[ ] [ ] [ ] CORE INIT [ ] [ ] [ ] "
# dump the global config as a debugging step
cat /cfg/${SYSTEM_NAME}/config.yaml

source /entry/project_env.sh

ping -c1 kameramaster

# this is a rolling counter just for fun, and also serves as something any client can always grab
REDIS_HOST=${REDIS_HOST:-nuvo0}
redis-client -h ${REDIS_HOST} incr term

# Seed Redis with the static system config before anything starts, so kamcore
# nodes (cam_param_monitor, etc.) read /sys/arch from Redis without depending on
# the GUI.
ros2 run kamcore seed_redis_config /cfg/${SYSTEM_NAME}/config.yaml

echo "Redis seeded. Core init complete; idling."
exec sleep infinity
