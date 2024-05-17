#!/usr/bin/env bash
## === === === ===  env configuration  === === === ===

# Not sure why this is necassary. Docker should already handle this.
source /entry/project_env.sh

## === === === ===  User-set configuration  === === === ===

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh  # get cq - ConfigQuery

## tell the system hostname of the node acting as ROS master host
export MASTER_HOST=$(cq '.master_host')

NODE_HOSTNAME=${NODE_HOSTNAME:-$(hostname)}
CAM_FOV=$(cq ".hosts.${NODE_HOSTNAME}.fov")
export NODE_HOSTNAME
export CAM_FOV

## === === === ===  Static configuration  === === === ===

WS_DEVEL="${KAM_REPO_DIR}/devel"
WS_VIAME=/root/kamera_ws
WS_KAMERA=/root/kamera_ws


# === === === === env part 1 === === === ===



for VNAME in CFG_ALIAS_SET
do
  if [[ -z "${!VNAME}" ]]
  then
    errcho "ERROR: Expected $VNAME environment variable that is missing."
    exit 1
  else
    echo "INFO: ENV ${VNAME} = ${!VNAME}"
    export "${VNAME}"
  fi
done

set_cam_ip() {
    # dev name is $1

    MAC=$(cq ".devices.${1}.mac")
    IP=$(cq ".devices.${1}.prefer_ip")
    #echo "$1: gevipconfig $MAC $IP 255.255.255.0"
    RES=$(gevipconfig -p "$MAC" "$IP" 255.255.255.0 2>/dev/null)
    RES2=$(	echo "${RES}" |  grep -o -Pe 'set to IP' )
    if [[ -n "$RES2"  ]]; then
	echo "$1 success! $RES"
    else
	echo "$1 failed: $RES"
    fi
}

declare -A PIDS
for DEV in $(cq  '.devices | keys | join("\n" )') ; do

    set_cam_ip "${DEV}" &
    PIDS[${DEV}]=$!
done

for pid in ${PIDS[*]}; do
    #printf '.'
    #echo "waiting on: $pid"
    wait $pid
done
