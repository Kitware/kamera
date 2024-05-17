#!/usr/bin/env bash

# Update /etc/hosts with our preconfigured ones
# /etc/hosts will tolerate multiple duplicate entries. It just takes the first row as the IP resolve

# $WS_DIR and $ROS_DISTRO should be provided by the container

# skip if we definitely know it's been appended with KAMERA entries
if [[ -n `grep KAMERA /etc/hosts` ]] ; then
    echo "Skipping /etc/hosts update, already contains KAMERA"
    exit 0
fi

echo "Adding custom host to /etc/hosts"
cp /etc/hosts /etc/orig.hosts
#cat /etc/orig.hosts $REPO_DIR/src/cfg/hosts > /etc/hosts
