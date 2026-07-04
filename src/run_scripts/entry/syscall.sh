#!/bin/bash

# System call service node

echo "<> <> <> SysCall <> <> <> "
source /entry/project.sh

exec ros2 launch sysinfo syscall.launch.xml \
    system_name:=`hostname`
