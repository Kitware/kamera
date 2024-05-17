#!/bin/bash

# System call service node

echo "<> <> <> SysCall <> <> <> "
source /entry/project.sh

ROSWAIT="--wait"

exec roslaunch "${ROSWAIT}" sysinfo syscall.launch \
    system_name:=`hostname` \
    verbosity:=$(/cfg/get ".verbosity")
