#!/bin/bash

# Camera parameter monitor node startup script

echo "<=> <=> <=>  FPS MONITOR  <=> <=> <=> "
source /entry/project.sh
source /aliases.sh

exec roslaunch --wait kamcore fps_monitor.launch norespawn:=${NORESPAWN:-false}
