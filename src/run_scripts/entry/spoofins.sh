#!/bin/bash

# INS node startup script with event spoofing

echo "( ) ( ) ( ) SPOOOOOOF INS ( ) ( ) ( ) "
source /entry/project.sh
source /aliases.sh
SPOOF_RATE=${SPOOF_INS} exec ros2 launch ins_driver ins.launch.xml