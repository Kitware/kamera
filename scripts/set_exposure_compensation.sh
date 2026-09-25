#!/bin/bash
# Allows you to change the auto exposure behavior of the cameras to either under or
# over exposure the imagery. A negative value under exposes the imagery, a postive
# value overexposes.
# Appropriate range seems to be in the single digits.
EXP_PARAM=0

SYSTEM_NAME=$(cat /home/user/kw/SYSTEM_NAME)
for host in center0${SYSTEM_NAME} left1${SYSTEM_NAME} right2${SYSTEM_NAME}; do
    docker exec -it cam-rgb-center bash -c "source /root/kamera/activate_ros.bash && rosservice call /${host}/rgb/rgb_driver/set_phaseone_parameter \"parameters: 'Exposure Comp.=${EXP_PARAM}'\""
done
