#!/bin/bash

#WSDIR=$(catkin locate)
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
WSDIR="$SCRIPTDIR/../../"
#SCRIPTDIR=$(catkin locate)/src/run_scripts/ee3/sentry

TMUX_SESSION=run_system_sim
DOCKER_NAME=roskamera

xhost +local:root

# Start tmux session
tmux new-session -d -s $TMUX_SESSION
tmux source ./.tmux.conf

# Roscore
tmux select-window -t $TMUX_SESSION:0
tmux rename-window -t $TMUX_SESSION:0 'Roscore'
#tmux send-keys "source ${SCRIPTDIR}/nuvo5k/000_core.sh" C-m
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make name=${DOCKER_NAME} rungui" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
#tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
tmux send-keys "roscore" C-m

sleep 5

# Image Simulator
tmux new-window -t $TMUX_SESSION:1 -n 'Image Sim'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make name=${DOCKER_NAME} rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
#tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
sleep 1
tmux send-keys "roslaunch --wait sensor_simulator simulate_cameras_one_sys.launch frame_rate:=1" C-m

sleep 1

# Image Nexus
tmux new-window -t $TMUX_SESSION:2 -n 'Image Nexus'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make name=${DOCKER_NAME} rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
#tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
sleep 1
tmux send-keys "roslaunch --wait image_nexus nexus.launch max_wait:=1" C-m

sleep 1

# Debayer
tmux new-window -t $TMUX_SESSION:3 -n 'DeBayer'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make name=${DOCKER_NAME} rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
#tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
sleep 1
tmux send-keys "roslaunch --wait color_processing debayer.launch" C-m

sleep 1

# INS
tmux new-window -t $TMUX_SESSION:4 -n 'INS Sim'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make name=${DOCKER_NAME} rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
#tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
sleep 1
tmux send-keys "roslaunch sensor_simulator simulate_ins.launch" C-m

sleep 1

# GUI
tmux new-window -t $TMUX_SESSION:5 -n 'GUI'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make name=${DOCKER_NAME} rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
#tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
sleep 1
tmux send-keys "roslaunch --wait wxpython_gui system_control_panel.launch" C-m

sleep 1

# ImageView
tmux new-window -t $TMUX_SESSION:6 -n 'ImageViewServer'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make name=${DOCKER_NAME} rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
#tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
sleep 1
tmux send-keys "roslaunch --wait wxpython_gui image_view_server.launch" C-m

sleep 1

# Test
tmux new-window -t $TMUX_SESSION:7 -n 'Test'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make name=${DOCKER_NAME} rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
#tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
sleep 1
tmux select-window -t $TMUX_SESSION:6
# Bring up the tmux session
tmux attach -t $TMUX_SESSION
