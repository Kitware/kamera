#!/bin/bash

# Camera parameter monitor node startup script

echo "<=> <=> <=>  FPS MONITOR  <=> <=> <=> "
source /entry/project.sh
source /aliases.sh

exec ros2 launch kamcore fps_monitor.launch.xml
