#!/bin/bash
# Creates a clock on the rgb camera port to sync the time to
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/../env.sh

/usr/sbin/ptp4l -f ${KAMERA_DIR}/src/cfg/uas/uas1/ptp/rgb.conf -i enp1s0
