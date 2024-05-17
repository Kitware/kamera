#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cam_id=devicemodul00_0f_31_02_c2_8d # RGB cam

gc_config "${cam_id}" "PtpMode=Slave"

rosrun rc_genicam_camera rc_genicam_camera _device:=$cam_id
