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



for VNAME in CFG_ALIAS_SET DATA_MOUNT_POINT
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

## grab params - HOPEFULLY these are set up
KAM_FLIGHT=$(rosparam get /flight -p)
KAM_PROJECT=$(rosparam get /project -p)
KAM_FLIGHT_DIR="${DATA_MOUNT_POINT}/${KAM_PROJECT}/fl${KAM_FLIGHT}"

echo "NODE_HOSTNAME: ${NODE_HOSTNAME}"
echo "NODE_MODE    : ${NODE_MODE}"
echo "DATA_MOUNT   : ${DATA_MOUNT_POINT}"
echo "KAM_FLIGHT   : ${KAM_FLIGHT}"
echo "KAM_PROJECT  : ${KAM_PROJECT}"
echo "_FLIGHT_DIR  : ${KAM_FLIGHT_DIR}"

for VNAME in KAM_FLIGHT KAM_PROJECT
do
  if [[ -z "${!VNAME}" ]]
  then
    errcho "ERROR: Expected $VNAME environment variable, could not talk to ROS master."
  else
    export "${VNAME}"
    GOOD=true
  fi
done

if [[ -z $GOOD ]]; then
    errcho "It seems the system is not running or cannot be contacted. You can enter the flight info manually"
    echo "=== Projects ==="
    ls "${DATA_MOUNT_POINT}/"
    echo "________________"
    read -p 'Project name: ' KAM_PROJECT
    echo "=== Flights ==="
    ls "${DATA_MOUNT_POINT}/${KAM_PROJECT}/"
    echo "________________"
    read -p 'Flight ## (include the "fl"): ' KAM_FLIGHT
    KAM_FLIGHT_DIR="${DATA_MOUNT_POINT}/${KAM_PROJECT}/${KAM_FLIGHT}"

fi
if [[ ! -d "$KAM_FLIGHT_DIR" ]]; then
    errcho "I could not find the directory: $KAM_FLIGHT_DIR. I will now exit."
    exit 1
fi


python -c "import sys; res = raw_input('Run post proc on $KAM_FLIGHT_DIR? (y/N) '); res = res or 'N'; sys.exit(int(str(res)[0].lower() !='y'))"

if [[ "$?" -ne 0 ]]; then
    errcho "Aborting"
    exit 0
fi

python /src/kamera/kamera/postflight/scripts/run_postflight.py \
    --summary \
    --detections \
    --geotiff \
    --verbosity 2 \
    --multi \
    --flight_dir "$KAM_FLIGHT_DIR"

PROC_RESULT="$?"
if [[ "${PROC_RESULT}" -ne 0 ]]; then
  rostopic pub --once /rawmsg std_msgs/String "data: 'ERROR [${PROC_RESULT}]: Postprocessing failed: ${KAM_FLIGHT_DIR}'"
else
  rostopic pub --once /rawmsg std_msgs/String "data: 'Geotiff processing complete ${KAM_FLIGHT_DIR}'"
fi
