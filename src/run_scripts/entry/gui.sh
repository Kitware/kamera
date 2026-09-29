#!/bin/bash

# INS node startup script

echo "GUI GUI GUI GUI GUI "

source /entry/project.sh
source /aliases.sh

for VNAME in SYSTEM_NAME ROS_MASTER_URI DATA_MOUNT_POINT
do
  if [[ -z "${!VNAME}" ]]
  then
    echo "ERROR: Expected $VNAME environment variable that is missing."
    exit 1
  else
    echo "INFO: ENV ${VNAME} = ${!VNAME}"
  fi
done

NODE_HOSTNAME=${NODE_HOSTNAME:-undefined}
#pip install --user "src/core/roskv/[redis]"
#catkin build custom_msgs roskv
#TODO TESTING CHANGES DON"T SAVE

if [[ -n "${START_IN_SHELL}" ]]; then
  bash
else
  exec roslaunch --wait wxpython_gui system_control_panel.launch
fi