#!/usr/bin/env bash

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
if [[ -z "${KAM_REPO_DIR}" ]]; then
    echo "ERROR: Could not resolve KAM_REPO_DIR. Check ~/.config/kamera"
    exit 1
fi

alias cb="catkin build"
alias nch="roslaunch"
alias pub="rostopic pub --once"
alias arc-off="set-is-archiving 0"
alias arc-on="set-is-archiving 1"
alias send-pulse="pub /daq/pulse std_msgs/UInt32 $1"
alias trig-freq="redis-cli -h $REDIS_HOST get /sys/arch/trigger_freq"
alias set-trig-freq="redis-cli -h $REDIS_HOST set /sys/arch/trigger_freq $1"
#alias set-trig-run="pub /daq/trigger_run std_msgs/Bool $1"
alias ips="ip -br addr"
alias scan="arp-scan 192.168.88.1/24"
alias tko="tmux kill-session"
alias rtls="rostopic list"
alias rnls="rosnode list"

alias kamwat="docker compose -f ${KAM_REPO_DIR}/compose/nodelist.yml run nodelist /entry/wat.sh"

setru() {
    export ROS_MASTER_URI=http://${1}:11311/
}

whoros() {
    echo $ROS_MASTER_URI
}

set-is-archiving() {
NODENAME=${2:-nuvo0}
rosservice call "/$NODENAME/set_archiving" "archiving: $1
project: 'bench'
flight: '7'
effort: 'test-effort'
notes: ''"
}


# catkin build shortcuts
alias cb-daq="catkin build custom_msgs mcc_daq"
alias cb-ins="catkin build custom_msgs ins_driver"
alias cb-cam="catkin build custom_msgs kamera_launch nexus kw_genicam_driver prosilica_camera"

alias cb-backend="catkin build backend"
alias cb-gui="catkin build wxpython_gui"


# runtime shortcuts
# core - only on center
run-core() {
    roslaunch kamcore kamcore.launch data_mount_point:="$DATA_MOUNT_POINT"
}
alias run1-core="run-core"

alias run-daq="roslaunch --wait mcc_daq daq.launch"
alias run-ins="roslaunch --wait ins_driver ins.launch"

# ins & daq
alias run2-daq="run-daq"
alias run3-ins="run-ins"

# todo: put this stuff in the .launch
# bring up both cams
run-rgb() {
    roslaunch --wait kamera_launch prosilica.launch ip:=${iprgb} system_name:=${NODE_HOSTNAME} trigger_mode:=syncin2
}
run-ir() {
    roslaunch --wait kw_genicam_driver genicam_a6750.launch camera_ipv4:=${ipir} \
        system_name:=${NODE_HOSTNAME} firmware_mode:=mono16 trigger_mode:=External
}


alias run4-rgb="run-rgb"
alias run5-ir="run-ir"


# bring up nexus
run-nexus() {
    roslaunch --wait nexus nexus.launch system_name:=${NODE_HOSTNAME}
}
alias run6-nexus="run-nexus"

run-debay() {
    roslaunch --wait color_processing debayer.launch system_name:=${NODE_HOSTNAME}
}
alias run7-debay="run-debay"


run-imageview() {
    roslaunch --wait wxpython_gui image_view_server.launch system_name:=${NODE_HOSTNAME}
}
alias run8-imageview="run-imageview"

alias run-gui="roslaunch --wait wxpython_gui system_control_panel.launch"



## ensure basic-aliases.sh for these
check-nuvos() {
    pingh mikrotik
    for i in 0 1 2; do pingh "nuvo$i" ; done
}

kill-rgb() {
    echo "rosnode kill /subsys${1}/rgb_driver"
    rosnode kill /subsys${1}/rgb/rgb_driver
}

kill-rgb-all() {
    for i in 2 1 0 ; do
        kill-rgb $i
    done;
}

kill-nodes() {

    nodes=$(rosnode list /subsys${1}/)
    rosnode kill ${nodes}
}

kill-nodes-all() {
    nodes=$(rosnode list /)
    rosnode kill ${nodes}
}

