#!/bin/sh

#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. /opt/ros/noetic/setup.sh

rosclean purge -y
mkdir -p ~/.tmuxinator
mkdir -p ~/.config/kamera/gui
