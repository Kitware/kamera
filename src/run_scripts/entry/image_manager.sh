#!/bin/bash

# Nexus node startup script

echo "/\ /\ /\   IMAGE VIEW    /\ /\ /\ "
source /entry/project.sh
source /aliases.sh

python3 /root/noaa_kamera/src/kitware-ros-pkg/image_manager/image_manager.py
