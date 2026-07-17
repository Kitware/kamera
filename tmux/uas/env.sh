#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export REDIS_HOST="uas0"

# Uncomment this line if you wish to run the GUI in "offline" mode
# (without uas0, 1, etc. hooked up)
# export REDIS_HOST="localhost"

_redis_elapsed=0
RESP=$(redis-cli -h "${REDIS_HOST}" ping 2>/dev/null)
while [ "$RESP" != "PONG" ]; do
    if [ $((_redis_elapsed % 30)) -eq 0 ]; then
        echo "Waiting for Redis at ${REDIS_HOST} (${_redis_elapsed}s elapsed, got '${RESP}')..."
    fi
    sleep 1
    _redis_elapsed=$((_redis_elapsed + 1))
    RESP=$(redis-cli -h "${REDIS_HOST}" ping 2>/dev/null)
done
unset _redis_elapsed

echo "Redis successfully connected at ${REDIS_HOST}, starting."

export NODE_HOSTNAME=$(hostname)
export ROS_MASTER_URI="http://${REDIS_HOST}:11311"
export KAMERA_DIR="/home/user/kw/kamera"
export DATA_MOUNT_POINT=$(redis-cli --raw -h ${REDIS_HOST} get /sys/arch/base)
export CAM_FOV=$(redis-cli --raw -h ${REDIS_HOST} get /sys/${NODE_HOSTNAME}/cam_fov)
export IR_DEVICE_ID=$(redis-cli --raw -h ${REDIS_HOST} get /sys/${NODE_HOSTNAME}/ir_device_id)
export RGB_DEVICE_ID=$(redis-cli --raw -h ${REDIS_HOST} get /sys/${NODE_HOSTNAME}/rgb_device_id)
