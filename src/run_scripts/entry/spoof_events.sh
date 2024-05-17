#!/bin/bash

# INS node startup script with event spoofing

echo "( ) ( ) ( ) SPOOOOOOF INS ( ) ( ) ( ) "
source /entry/project.sh
roslaunch --wait ins_driver spoof_events.launch
