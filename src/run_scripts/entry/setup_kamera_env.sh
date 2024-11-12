#!/usr/bin/env bash

## TODO:     IN THE PROCESS OF DEPRECATION IN FAVOR OF yq-based config

# Set up the environmental variables that define the behavior of the current node
# This is useful both inside and out of containers

## === === === ===  User-set configuration  === === === ===

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh  # get cq - ConfigQuery

## tell the system hostname of the node acting as ROS master host
export MASTER_HOST=$(cq '.master_host')

## === === === ===  Static configuration  === === === ===

export _ROS_PORT=11311


## === === === ===  default values === === ===
# index number of the system, 0 being master
SYSIDX=0


CAM_FOV=${CAM_FOV:-NONE}


## === === === ===  Infer variables === === ===
# Determine env variables based on configuration
# and parameters from system such as hostname


# Check cgroup to see if we are running inside a container
if [[ -n `grep docker /proc/1/cgroup` ]] ; then
    export IS_CONTAINER=true
else
    export REPO_DIR=$HOME/kw/kamera
fi

## name of current host
export HOST=`hostname`

## name of current host, passed to ROS
export ROS_HOSTNAME=`hostname`

NODE_HOSTNAME=${NODE_HOSTNAME:-${ROS_HOSTNAME}}

echo "node hostname: $NODE_HOSTNAME . is_container: ${IS_CONTAINER:-false}"

if [[ -z ${NODE_HOSTNAME} ]] ; then
    echo "WARNING: NODE_HOSTNAME not set! This happens in builds, but should not happen in production"
fi

if [[ -z $IS_CONTAINER ]] ; then
    SUBDIR=/kw
    export REPO_DIR=${HOME}${SUBDIR}/kamera
    export WS_DIR=${HOME}${SUBDIR}/kamera_ws
fi


## This is some jankitude to set the pod. I really wanna fix this
export NODE_HOSTNAME
export ROS_MASTER_URI="http://${MASTER_HOST}:${_ROS_PORT}/"
