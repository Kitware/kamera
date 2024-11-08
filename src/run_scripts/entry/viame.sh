#!/bin/bash

## === === === ===  User-set configuration  === === === ===

source /entry/project_env.sh
source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh  # get cq - ConfigQuery

## tell the system hostname of the node acting as ROS master host
export MASTER_HOST=$(cq '.master_host')
export REDIS_HOST=$(cq '.redis_host')

NODE_HOSTNAME=${NODE_HOSTNAME:-$(hostname)}
CAM_FOV="$(redis-cli -h ${REDIS_HOST} -p 6379 get "/sys/arch/hosts/${NODE_HOSTNAME}/fov")"
CAM_FOV=${CAM_FOV:-"null_fov"}
FOV_SHORT=`echo $CAM_FOV | cut -c1-1`
FOV_SHORT="${FOV_SHORT^}"
export NODE_HOSTNAME
export CAM_FOV
export FOV_SHORT

## === === === ===  Static configuration  === === === ===

WS_DEVEL="${KAM_REPO_DIR}/devel"


## === === === ===  catkin rebuild  === === === ===

if [[ $(redis-cli --raw -h $REDIS_HOST get /debug/rebuild ) == "true" ]]; then
  echo "/debug/rebuild set, triggering rebuild on startup"
  # todo: remove this shim
  #  we have to clean first since the current docker image puts roskv in the wrong spot
  catkin clean -y
  catkin build sprokit_adapters
  if [[ $? -ne 0 ]]; then
    echo "Rebuild failed. Your code is in an unstable state"
    exit 1
  else
    echo "Rebuild succeeded"
  fi
else
  echo " ~~~ skipping rebuild ~~~ "
fi

# === === === === env part 1 === === === ===

# This pulls in VIAME_INSTALL variable
source "${KAM_REPO_DIR}/src/run_scripts/setup/setup_viame_runtime.sh"


for VNAME in CFG_ALIAS_SET CAM_FOV MASTER_HOST WS_DEVEL VIAME_INSTALL
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


# Check gpu
nvidia-smi
if [[ $? -ne 0 ]]; then
    errcho "ERROR: nvidia-smi failed. Check GPU/runtime"
    exit 1
fi

# Directory to store detection csv files.
_DETECTION_CSV_DIR="${DATA_MOUNT_POINT}/$(date +%Y%m%d%H%M%S)"

PIPEFILE=$(redis-cli -h ${REDIS_HOST} -p 6379 get "/sys/${NODE_HOSTNAME}/detector/pipefile")
DETECTION_CSV_DIR="$(redis-cli -h ${REDIS_HOST} -p 6379 get "/sys/syscfg_dir")/../detections"

DETECTION_DSV_DIR=${DETECTION_DSV_DIR:-$_DETECTION_CSV_DIR}
# Directory to store detection image list files.
IMAGE_LIST_DIR=$DETECTION_CSV_DIR

mkdir -p ${DETECTION_CSV_DIR}
mkdir -p ${IMAGE_LIST_DIR}

# Defaults
_PIPEFILE="/mnt/flight_data/detector_models/viame-configs/pipelines/embedded_dual_stream/EO_Seal_Detector.pipe"
PIPEFILE=${PIPEFILE:-$_PIPEFILE}
PIPELINE_DIR=$(dirname $PIPEFILE)

for VNAME in PIPEFILE KWIVER_PLUGIN_PATH
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

if [[ ! -f "${PIPEFILE}" ]]; then
    errcho "ERROR: Pipefile not found: $PIPEFILE"
    exit 1
fi

if [[ $(redis-cli --raw -h $REDIS_HOST get /sys/detector/read_from_nas ) == "true" ]]; then
  echo "Reading from NAS enabled, increasing queue size to 15000."
  SYNC_Q_SIZE=15000
else
  echo "Reading from NAS disabled, decreasing queue size to 1."
  SYNC_Q_SIZE=1
fi

## === === === ===  end configuration  === === === ===

## grab params - HOPEFULLY these are set up
#TODO Pull from Redis
export KAM_FLIGHT="$(redis-cli -h ${REDIS_HOST} -p 6379 get "/sys/arch/flight")"

echo "NODE_HOSTNAME: ${NODE_HOSTNAME}"
echo "NODE_MODE    : ${NODE_MODE}"
echo "DATA_MOUNT   : ${DATA_MOUNT_POINT}"
echo "DETN_CSV_DIR : ${DETECTION_CSV_DIR}"
echo "KAM_FLIGHT   : ${KAM_FLIGHT}"

## === === === ===  env configuration  === === === ===

echo "PIPEFILE that's in use: ${PIPEFILE}"
printf "
\$ exec roslaunch sprokit_adapters sprokit_detector_fusion_adapter.launch \
                    kwiver:=${WS_DEVEL} \
                    system_name:=${NODE_HOSTNAME} \
                    detector_node:=detector \
                    detection_pipefile:="${PIPEFILE}" \
                    embed_det_chips:=true \
                    pad_det_chip_percent:=50 \
                    det_topic:=/${NODE_HOSTNAME}/detections \
                    detector_id_string:=seal \
                    synchronized_images_in1:=/${NODE_HOSTNAME}/synched \
                    rgb_port_ind:=1 \
                    ir_port_ind:=2 \
                    uv_port_ind:=0 \
                    redis_uri:=tcp://192.168.88.100:6379 \
                    sync_q_size:=${SYNC_Q_SIZE} \
                    detection_csv_dir:=${DETECTION_CSV_DIR} \
                    norespawn:=${NORESPAWN:-false} \
                    image_list_dir:=${IMAGE_LIST_DIR}
                    "
# TODO:
# Hack because I forgot this in the built image
rosparam set /${NODE_HOSTNAME}/detector/sync_q_size ${SYNC_Q_SIZE}

# Use the fork syntax so we can catch if roslaunch exits (since it only exits 0)
# Manually catch failures here so at least we have the option of letting docker restart
roslaunch --wait sprokit_adapters sprokit_detector_fusion_adapter.launch \
                    kwiver:=${WS_DEVEL} \
                    system_name:=${NODE_HOSTNAME} \
                    detector_node:=detector \
                    detection_pipefile:="${PIPEFILE}" \
                    pipeline_dir:="${PIPELINE_DIR}" \
                    embed_det_chips:=true \
                    pad_det_chip_percent:=50 \
                    det_topic:=/${NODE_HOSTNAME}/detections \
                    detector_id_string:=seal \
                    synchronized_images_in1:=/${NODE_HOSTNAME}/synched \
                    uv_port_ind:=0 \
                    rgb_port_ind:=1 \
                    ir_port_ind:=2 \
                    redis_uri:="tcp://192.168.88.100:6379" \
                    ocv_num_threads:=4 \
                    sync_q_size:=${SYNC_Q_SIZE} \
                    detection_csv_dir:=${DETECTION_CSV_DIR} \
                    norespawn:=${NORESPAWN:-false} \
                    image_list_dir:=${IMAGE_LIST_DIR} 2> >(tee -a /tmp/roslaunch_err.log >&2) &
STAT_ROS=$!
wait $STAT_ROS
echo "roslaunch probably died with a 0 error code"
RES=$(grep -Po -e 'REQUIRED.+ has died' /tmp/roslaunch_err.log)
if [[ -n $RES ]]; then
   echo $RES
   exit 1
fi
