#!/bin/bash

# INS node startup script

echo "( ) ( ) ( ) INS ( ) ( ) ( ) "
source /entry/project.sh
source /aliases.sh

# Spoofing is controlled via the SPOOF_RATE / SPOOF_INS environment variables
RESPAWN=$([[ "${NORESPAWN}" == "true" ]] && echo false || echo true)
exec ros2 launch ins_driver ins.launch.xml respawn:=${RESPAWN}
