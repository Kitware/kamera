#!/bin/bash
# Sets shutter to ES mode

SYSTEM_NAME=$(cat /home/user/kw/SYSTEM_NAME)
for host in center0${SYSTEM_NAME} left1${SYSTEM_NAME} right2${SYSTEM_NAME}; do
    docker exec -it cam-rgb-center bash -c "source /root/kamera/activate_ros.bash && rosservice call /${host}/rgb/rgb_driver/set_phaseone_parameter \"parameters: 'Shutter Mode=2'\""
done
