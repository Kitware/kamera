#!/usr/bin/env bash


# Image transport plugins create a bunch of unnecssary topics. We are gonna
# force-uninstall them to keep the clutter down
printf "Force-uninstall transport plugins which come bundled with ros-desktop-full.
This cleans up the topic-space in ROS.
This will show some angry looking messages.
Do not use this script if you actually need these message channels.
"
dpkg -r --force-depends \
    ros-kinetic-theora-image-transport \
    ros-kinetic-compressed-depth-image-transport \
    ros-kinetic-compressed-image-transport \
    ros-kinetic-image-transport-plugins