#!/bin/bash
SYSTEM_NAME="$(cat ${HOME}/kw/SYSTEM_NAME)"
export SYSTEM_NAME

export KAMERA_DIR=$(${HOME}/.config/kamera/repo_dir.bash)
CFG_FILE="${KAMERA_DIR}/src/cfg/${SYSTEM_NAME}/config.yaml"
cq () {
    CFG_FILE=${CFG_FILE} ${KAMERA_DIR}/src/cfg/get "$@"
}

export REDIS_HOST=$(cq ".redis_host")

# Uncomment this line if you wish to run the GUI in "offline" mode
# (without nuvo0, 1, etc. hooked up)
# export REDIS_HOST="localhost"

RESP=$(redis-cli -h ${REDIS_HOST} ping)
while [ "$RESP" != "PONG" ]
do
    echo "Got '$RESP', wanted 'PONG'."
    echo "Waiting for redis host $REDIS_HOST to come online..."
    RESP=$(redis-cli -h ${REDIS_HOST} ping)
    sleep 1;
done

echo "Redis successfully connected at $REDIS_HOST, starting."

export ROS_HOSTNAME=$(cq ".master_host")
export NODE_HOSTNAME=$(hostname)
export ROS_MASTER_URI="http://${ROS_HOSTNAME}:11311"
export DOCKER_KAMERA_DIR="/root/kamera"
export DATA_MOUNT_POINT=$(redis-cli --raw -h ${REDIS_HOST} get /sys/arch/base)
export CAM_FOV=$(redis-cli --raw -h ${REDIS_HOST} get /sys/arch/hosts/${NODE_HOSTNAME}/fov)

export ROS_DISTRO="noetic"
export KAMERA_DNS_IP="192.168.88.1"
export PULSE_TTY=/dev/ttyS0
export MCC_DAQ="/dev/$(readlink /dev/mcc_daq)"

# Toggles reading detector images from NAS vs. reading from ROS msg
# INCREASED I/O
export READ_FROM_NAS=0

# Toggles the option to compress / decompress images within the nexus
# To test the detector's performance on compressed imagery
# INCREASED LATENCY (~0.7s)
export COMPRESS_IMAGERY=0

# Sets the compression used on the phase on imagery
# Best to keep within 80-100 for optimal quality
export JPEG_QUALITY=90

# If in a GPS-denied environment, this will publish events on each event
# trigger to simulated the GPS timestamp
# BE SURE TO SET TO 0 BEFORE FLIGHT
export SPOOF_EVENTS=1
