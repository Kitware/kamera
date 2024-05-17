#!/bin/bash

# Nexus node startup script

echo "<> <> <> NEXUS <> <> <> "
source /entry/project.sh

pip install redis

ROSWAIT="--wait"

exec roslaunch "${ROSWAIT}" nexus nexus.launch \
    system_name:=${NODE_HOSTNAME} \
    verbosity:=$(/cfg/get ".verbosity")
