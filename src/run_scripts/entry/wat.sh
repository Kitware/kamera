#!/bin/bash

# Diagnostics

errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

echo "[?] [?] [?]  WAT. [?] [?] [?]  "

source /entry/project_env.sh

# Expected exit code from a Ctrl-C when in explicit docker run mode.
trap "errcho 'Caught SIGINT'; cleanup" SIGINT
# Expected exit code from docker stop command.
trap "errcho 'Caught SIGTERM'; cleanup" SIGTERM

echo "=== === === === ROS2 environment === === === === "
echo "ROS_DISTRO      : ${ROS_DISTRO}"
echo "ROS_DOMAIN_ID   : ${ROS_DOMAIN_ID}"
echo "RMW_IMPLEMENTATION: ${RMW_IMPLEMENTATION:-default}"

echo "=== === === === /etc/hosts: === === === === "
cat /etc/hosts

echo "=== === === === /etc/resolv.conf: === === === === "
cat /etc/resolv.conf

echo "=== === === === visible nodes === === === === "
ros2 node list || true

echo "=== === === === visible topics === === === === "
ros2 topic list || true

exec ros2 doctor --report
