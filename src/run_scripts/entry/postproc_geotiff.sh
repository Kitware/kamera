#!/usr/bin/env bash
## === === === ===  env configuration  === === === ===

export KAM_FLIGHT_DIR=$(redis-cli --raw -h ${REDIS_HOST} get /postproc/${NODE_HOSTNAME}/geotiff/flight_dir)

## grab params - HOPEFULLY these are set up

echo "FIRST COMMAND"
echo $1
echo "CAM_FOV: ${CAM_FOV}"
echo "NODE_HOSTNAME: ${NODE_HOSTNAME}"
echo "DATA_MOUNT   : ${DATA_MOUNT_POINT}"
echo "KAM_FLIGHT_DIR  : ${KAM_FLIGHT_DIR}"

redis-cli -h ${REDIS_HOST} set "/postproc/${NODE_HOSTNAME}/geotiff/busy" 1
python /src/kamera/kamera/postflight/scripts/run_postflight.py \
    --verbosity 2 \
    --multi \
    --geotiff \
    --flight_dir "$KAM_FLIGHT_DIR"
PROC_RESULT="$?"
redis-cli -h ${REDIS_HOST} set "/postproc/${NODE_HOSTNAME}/geotiff/busy" 0

if [[ "${PROC_RESULT}" -ne 0 ]]; then
  redis-cli -h ${REDIS_HOST} set "/postproc/${NODE_HOSTNAME}/geotiff/status" "ERROR [${PROC_RESULT}]: Postprocessing failed: ${KAM_FLIGHT_DIR}"
else
  redis-cli -h ${REDIS_HOST} set "/postproc/${NODE_HOSTNAME}/geotiff/status" "Success: Flight Summary complete ${KAM_FLIGHT_DIR}"
fi

