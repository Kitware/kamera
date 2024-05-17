#!/usr/bin/env bash
## === === === ===  env configuration  === === === ===

# Not sure why this is necassary. Docker should already handle this.
source /entry/env.sh
export KAM_FLIGHT_DIR=$(redis-cli --raw -h ${REDIS_HOST} get /postproc/${NODE_HOSTNAME}/detections/flight_dir)

## grab params - HOPEFULLY these are set up

echo "FIRST COMMAND"
echo $1
echo "CAM_FOV: ${CAM_FOV}"
echo "NODE_HOSTNAME: ${NODE_HOSTNAME}"
echo "DATA_MOUNT   : ${DATA_MOUNT_POINT}"
echo "KAM_FLIGHT_DIR  : ${KAM_FLIGHT_DIR}"

redis-cli -h ${REDIS_HOST} set "/postproc/${NODE_HOSTNAME}/detections/busy" 1
python /src/kamera/kamera/postflight/scripts/run_postflight.py \
    --detections \
    --verbosity 2 \
    --multi \
    --flight_dir "$KAM_FLIGHT_DIR"
PROC_RESULT="$?"
redis-cli -h ${REDIS_HOST} set "/postproc/${NODE_HOSTNAME}/detections/busy" 0

if [[ "${PROC_RESULT}" -ne 0 ]]; then
  redis-cli -h ${REDIS_HOST} set "/postproc/${NODE_HOSTNAME}/detections/status" "ERROR [${PROC_RESULT}]: Detection tuning failed: ${KAM_FLIGHT_DIR}"
else
  redis-cli -h ${REDIS_HOST} set "/postproc/${NODE_HOSTNAME}/detections/status" "Success: Detection Tuning Completed ${KAM_FLIGHT_DIR}"
fi
