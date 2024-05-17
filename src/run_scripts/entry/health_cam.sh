#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
if [[ -z "${CAM_MODE}" ]]; then
  (>&2 echo "CAM_MODE is not defined")
  exit 1
fi
if [[ -z "${HOSTNAME}" ]]; then
  (>&2 echo "HOSTNAME is not defined")
  exit 1
fi

rosservice call "/${HOSTNAME}/${CAM_MODE}/${CAM_MODE}_driver/health" "{}"