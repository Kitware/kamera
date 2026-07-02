#!/bin/bash

# Camera parameter monitor node startup script

echo "<=> <=> <=>  CAM PARAM MONITOR  <=> <=> <=> "
source /entry/project.sh
source /aliases.sh

exec ros2 launch kamcore shapefile_monitor.launch.xml
