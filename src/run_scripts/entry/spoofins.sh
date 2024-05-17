#!/bin/bash

# INS node startup script with event spoofing

echo "( ) ( ) ( ) SPOOOOOOF INS ( ) ( ) ( ) "
source /entry/project.sh
source /aliases.sh
roslaunch --wait ins_driver ins.launch spoof:=${SPOOF_INS}