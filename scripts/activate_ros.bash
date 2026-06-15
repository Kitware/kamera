#!/usr/bin/env bash
#
# Activate the ROS bash environment, either sourcing the current workspace
# devel setup or the base ROS environment setup script.  This cascades down
# ROS versions when a development environment does not exist.

# may have changed
MASTER_HOST=$(echo $ROS_MASTER_URI| grep -Po -e '(?<=http:\/\/)([\w\.]+)(?=:)')

rosinfo () {
printf "=== === === ACTIVATE ROS ${ROS_DISTRO} === === === === ===
DATA_MOUNT_POINT: ${DATA_MOUNT_POINT}
CAM_FOV         : ${CAM_FOV}
HOSTNAME        : `hostname`
NODE_HOSTNAME   : ${NODE_HOSTNAME}
ROS_HOST/IP     : ${ROS_HOSTNAME} ${ROS_IP}
ROS_MASTER_URI  : ${ROS_MASTER_URI}
REPO_DIR        : ${REPO_DIR}
"
}


rosinfo

echo "Sourcing files and establishing environment"
## This presumes ROS_DISTRO is set
DEVEL_SETUP="${REPO_DIR}/devel/setup.bash"
VERSION_SETUP_PATH="/opt/ros/${ROS_DISTRO}/setup.bash"
if [ -f "${VERSION_SETUP_PATH}" ]
then
  echo "Sourcing ROS setup version: ${ROS_DISTRO}: ${VERSION_SETUP_PATH}"
  source "${VERSION_SETUP_PATH}"
else
# No version setup script found. Report that.
echo "ERROR: Found no ROS setup script for considered ROS versions: ${ROS_DISTRO}"
fi

if [ -f "${DEVEL_SETUP}" ]
then
  echo "Sourcing workspace devel setup"
  source "${DEVEL_SETUP}"
else
  echo "WARNING: Found no ROS devel setup script in workspace: ${DEVEL_SETUP}"
fi

### ok ros devel/setup.bash does some weird stuff with path so we have to make sure it's still set up correctly
## easier to not add duplicates than remove duplicates

## alternate syntax
#[ "${PATH#*$HOME/.local/bin:}" == "$PATH" ] && export PATH="$HOME/.local/bin:$PATH"
#[ "${PATH#*$REPO_DIR/src/run_scripts/inpath:}" == "$PATH" ] && export PATH="$REPO_DIR/src/run_scripts/inpath:$PATH"
#[ "${PATH#*/opt/ros/$ROS_DISTRO/bin:}" == "$PATH" ] && export PATH="/opt/ros/$ROS_DISTRO/bin:$PATH"

echo "activate_ros::PATH: [${PATH}]"

# this sets the logging format
export ROSCONSOLE_FORMAT='${walltime}: ${message}'
