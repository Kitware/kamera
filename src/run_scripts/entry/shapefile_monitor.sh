#!/bin/bash

# Camera parameter monitor node startup script

echo "<=> <=> <=>  CAM PARAM MONITOR  <=> <=> <=> "
source /entry/project.sh
source /aliases.sh

exec roslaunch --wait kamcore shapefile_monitor.launch norespawn:=${NORESPAWN:-false}
