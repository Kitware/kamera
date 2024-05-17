#!/bin/bash

# Nexus node startup script

echo "~ ~ ~ ~ ~   IR   ~ ~ ~ ~ ~  "
source /entry/project_env.sh

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh  # get cq - ConfigQuery

for VNAME in CFG_ALIAS_SET CAM_FOV CAM_MODE
do
  if [[ -z "${!VNAME}" ]]
  then
    echo "ERROR: Expected $VNAME environment variable that is missing."
    exit 1
  else
    echo "INFO: ENV ${VNAME} = ${!VNAME}"
  fi
done

NODE_HOSTNAME=${NODE_HOSTNAME:-undefined}
DEV_ID=$(cq ".locations.${CAM_FOV}.${CAM_MODE}")
if [[ ${DEV_ID} == 'null' ]]; then
    printf "<!> Not a valid camera FOV/ mode: ${CAM_FOV}:${CAM_MODE}. oneof: {ir}"
    exit 1
fi

GUID=$(/cfg/get ".devices.${DEV_ID}.mac")

if [[ ${GUID} == 'null' ]]; then
    printf "<!> Unable to find camera GUID: ${CAM_FOV}:${CAM_MODE}"
    exit 1
fi

CAM_IP=$(cq ".devices.${DEV_ID}.prefer_ip")

if [[ ${CAM_IP} == 'null' ]]; then
    printf "<!> Unable to find camera prefer_ip: ${CAM_FOV}:${CAM_MODE}"
    exit 1
fi



ROSWAIT="--wait"
CAM_PIXEL_FORMAT=${CAM_PIXEL_FORMAT:-mono16}
CAM_TRIGGER_SOURCE=${CAM_TRIGGER_SOURCE:-External}
CAM_TIMEOUT=${CAM_TIMEOUT:-3333}
DRIVER=$(cq ".devices.${DEV_ID}.model").launch
# extra arguments to pass to roslaunch in the form of `argname1:=val argname2:=val`
CAM_EXTRA_ARGS=${CAM_EXTRA_ARGS:-}
LOGFILE="/tmp/roslaunch_err_${CAM_FOV}_${CAM_MODE}.log"

printf "\e[32m
DEV_ID      : ${DEV_ID}
GUID (MAC)  : ${GUID}
CAM_IP      : ${CAM_IP}
DRIVER      : ${DRIVER}
EXTRA ARGS  : ${CAM_EXTRA_ARGS}
\e[0m
"
if [[ $(redis-cli --raw -h $REDIS_HOST get /debug/rebuild ) == "true" ]]; then
  echo "/debug/rebuild set, triggering rebuild on startup"
  catkin build kw_genicam_driver
  if [[ $? -ne 0 ]]; then
    echo "Rebuild failed. Your code is in an unstable state"
    exit 1
  else
    echo "Rebuild succeeded"
  fi
fi

exec roslaunch "${ROSWAIT}" kw_genicam_driver ${DRIVER} \
    system_name:=${NODE_HOSTNAME} \
    cam_fov:=${CAM_FOV} \
    camera_ipv4:=${CAM_IP} \
    camera_manufacturer:=FLIR \
    firmware_mode:=${CAM_PIXEL_FORMAT} \
    nextImage_timeout:=${CAM_TIMEOUT} \
    info_verbosity:=$(/cfg/get ".verbosity") \
    ${CAM_EXTRA_ARGS} 2> >(tee -a "${LOGFILE}" >&2) &

STAT_ROS=$!
wait $STAT_ROS
echo "roslaunch probably died with a 0 error code"
RES=$(grep -Po -e 'REQUIRED.+ has died' "${LOGFILE}")
if [[ -n $RES ]]; then
   echo $RES
   exit 1
fi
