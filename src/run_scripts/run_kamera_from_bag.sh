#!/bin/bash

#WSDIR=$(catkin locate)
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
WSDIR="$SCRIPTDIR/../../"

TMUX_SESSION=run_system_from_bag

# Specify the bag to replay.
BAGFNAME='/mnt/2019-02-12-03-48-12.bag'

tmux kill-session -t $TMUX_SESSION

# Start tmux session
tmux new-session -d -s $TMUX_SESSION
#tmux source ./.tmux.conf

# Roscore
tmux select-window -t $TMUX_SESSION:0
tmux rename-window -t $TMUX_SESSION:0 'Roscore'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "docker compose up -d kamera && docker exec -it kamera bash" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
tmux send-keys "roscore" C-m

sleep 3

# Image driver
tmux new-window -t $TMUX_SESSION:1 -n 'Rosbag'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
tmux send-keys "rosbag play ${BAGFNAME} --l" C-m

# Image Nexus
tmux new-window -t $TMUX_SESSION:2 -n 'Image Nexus'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
tmux send-keys "catkin build nexus" C-m
tmux send-keys "data_mount_point=/media/mattb/datapartition/kamera/testing" C-m
tmux send-keys "roslaunch --wait nexus nexus.launch max_wait:=1" C-m

# Debayer
tmux new-window -t $TMUX_SESSION:3 -n 'DeBayer'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
tmux send-keys "catkin build color_processing" C-m
tmux send-keys "roslaunch --wait color_processing debayer.launch" C-m

# INS
tmux new-window -t $TMUX_SESSION:4 -n 'INS Sim'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
tmux send-keys "catkin build sensor_simulator" C-m
tmux send-keys "source activate_ros.bash" C-m
tmux send-keys "roslaunch sensor_simulator simulate_ins.launch \
                lat:=67.32755634 lon:=-166.53534026 height:=300" C-m

# GUI
tmux new-window -t $TMUX_SESSION:5 -n 'GUI'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "source run_gui.sh" C-m
tmux send-keys "roslaunch --wait wxpython_gui system_control_panel.launch" C-m

# ImageView
tmux new-window -t $TMUX_SESSION:6 -n 'ImageViewServer'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
tmux send-keys "roslaunch --wait wxpython_gui image_view_server.launch" C-m

# Test
tmux new-window -t $TMUX_SESSION:7 -n 'Test'
tmux send-keys "cd ${WSDIR}" C-m
tmux send-keys "make rungui-existing" C-m
tmux send-keys "source /opt/ros/kinetic/setup.bash" C-m
tmux send-keys "source activate_ros.bash" C-m
tmux select-window -t $TMUX_SESSION:6
# Bring up the tmux session
tmux attach -t $TMUX_SESSION
