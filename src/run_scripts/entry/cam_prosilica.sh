#!/bin/bash

# Cam driver node startup script
# receive this from the channel-specific launcher. Having trouble with positional args
# cause of the project entrypoint.
#CAM_MODE=${CAM_MODE:-rgb}
CAM_MODE=${1}
# todo: un-hardcode this
export ARCH_KEY="/src/arch"
TRIGGER_MODE=${TRIGGER_MODE:-syncin2}
NODE_HOSTNAME=${NODE_HOSTNAME:-$(hostname)}

source /entry/project_env.sh

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)


if [[ $(redis-cli --raw -h $REDIS_HOST get /debug/enable ) == "true" ]]; then
  export KAMERA_DEBUG=true
fi


CAM_FOV=${CAM_FOV:-$(jq -r ".arch.hosts[\"${NODE_HOSTNAME}\"].fov" "$KAMERA_CFG")}


for VNAME in SYSTEM_NAME CAM_FOV CAM_MODE
do
  if [[ -z "${!VNAME}" ]]; then
    echo "ERROR: Expected $VNAME environment variable that is missing."
    exit 1
  elif [[ "${!VNAME}" == 'null' ]] ; then
    echo "ERROR: $VNAME is null. check config"
    exit 1
  else
    echo "INFO: ENV ${VNAME} = ${!VNAME}"
  fi
done

echo "% % % %    CAM-${CAM_MODE}-${CAM_FOV}    % % % %"

DEV_ID=$(jq -r ".locations.${CAM_FOV}.${CAM_MODE}" "$KAMERA_CFG")
if [[ ${DEV_ID} == 'null' ]]; then
    printf "<!> Not a valid camera FOV/ mode: ${CAM_FOV}:${CAM_MODE}. oneof: {ir}"
    exit 1
fi

CAM_IFACE=$(jq -r ".interfaces.${CAM_MODE}" "$KAMERA_CFG")
if [[ ${CAM_IFACE} == 'null' ]]; then
    printf "<!> Not a valid camera FOV/ mode: ${CAM_FOV}/${CAM_MODE}.
    CAM_FOV: oneof: `jq -r '.locations | keys' "$KAMERA_CFG" | tr '\n' ' '`
    CAM_MODE: oneof: `jq -r '.channels ' "$KAMERA_CFG" | tr '\n' ' '`\n"
    exit 1
fi

IFACE_INFO=$(${KAM_REPO_DIR}/src/run_scripts/inpath/ipj.sh | jq ".$CAM_IFACE")
if [[ ${CAM_IFACE} == 'null' ]]; then
    printf "<!> Failed to get info for interface $CAM_IFACE"
    exit 1
fi

#CAM_IP=$(locate-attached.sh "$CAM_IFACE")
CAM_IP=$(jq -r ".devices.${DEV_ID}.prefer_ip" "$KAMERA_CFG")

if [[ $? -ne 0 || ${CAM_IP} == 'null' ]]; then
    printf "<!> Unable to find camera IP: ${CAM_FOV}:${CAM_MODE} on interface ${CAM_IFACE}"
    exit 1
fi

printf "
NODE_HN : ${NODE_HOSTNAME}
IFACE   : ${CAM_IFACE}
CAM_IP  : ${CAM_IP}
"

function cleanup {

  pkill -2 prosilica_nodelet
}

# Expected exit code from a Ctrl-C when in explicit docker run mode.
trap "errcho 'Caught SIGINT'; cleanup" SIGINT
# Expected exit code from docker stop command.
trap "errcho 'Caught SIGTERM'; cleanup" SIGTERM

ROSWAIT="--wait"
LOGFILE="/tmp/roslaunch_err_${CAM_FOV}_${CAM_MODE}.log"

if [[ $(redis-cli --raw -h $REDIS_HOST get /debug/rebuild ) == "true" ]]; then
  echo "/debug/rebuild set, triggering rebuild on startup"
  catkin build prosilica_camera
  if [[ $? -ne 0 ]]; then
    echo "Rebuild failed. Your code is in an unstable state"
    exit 1
  else
    echo "Rebuild succeeded"
  fi
fi

exec roslaunch "${ROSWAIT}" prosilica_camera prosilica.launch \
    ip:=${CAM_IP} \
    system_name:=${NODE_HOSTNAME} \
    cameratype:=${CAM_MODE} \
    cam_fov:=${CAM_FOV} \
    trigger_mode:=${TRIGGER_MODE} \
    norespawn:=${NORESPAWN} \
    GainMode:=$(jq -r ".launch.cam.${CAM_MODE}.GainMode" "$KAMERA_CFG") \
    GainValue:=$(jq -r ".launch.cam.${CAM_MODE}.GainValue" "$KAMERA_CFG") 2> >(tee -a "${LOGFILE}" >&2) &

STAT_ROS=$!
wait $STAT_ROS
echo "roslaunch probably died with a 0 error code"
RES=$(grep -Po -e 'REQUIRED.+ has died' "${LOGFILE}")
if [[ -n $RES ]]; then
   echo $RES
   exit 1
fi
