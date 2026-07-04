#!/bin/bash

# INS node startup script

echo "GUI GUI GUI GUI GUI "


KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh  # get cq - ConfigQuery


source /entry/project.sh
source /aliases.sh

for VNAME in CFG_ALIAS_SET ROS_DOMAIN_ID DATA_MOUNT_POINT
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

if [[ -n "${START_IN_SHELL}" ]]; then
  bash
else
  exec ros2 launch wxpython_gui system_control_panel.launch.xml
fi