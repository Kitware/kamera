#!/bin/bash

# Nexus node startup script

echo "? @ ?    DeBAYERING   ? @ ? "
source /entry/project.sh
source /aliases.sh
NODE_HOSTNAME=${NODE_HOSTNAME:-undefined}
roslaunch --wait color_processing debayer.launch system_name:=${NODE_HOSTNAME}
