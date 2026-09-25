#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cam_id=ircam0 # IR cam

#gc_config "${cam_id}" "PtpMode=Auto"

export ROS_NAMESPACE="/ir"
rosrun rc_genicam_camera rc_genicam_camera _device:=$cam_id
