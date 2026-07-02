#!/bin/bash

# INS node startup script with event spoofing

echo "( ) ( ) ( ) SPOOOOOOF INS ( ) ( ) ( ) "
source /entry/project.sh
ros2 launch ins_driver spoof_events.launch.xml
