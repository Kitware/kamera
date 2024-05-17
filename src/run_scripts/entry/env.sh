#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export REDIS_HOST="cas0"

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

export NODE_HOSTNAME=$(hostname)
export ROS_MASTER_URI="http://${REDIS_HOST}:11311"
export KAMERA_DIR="/home/user/kw/noaa_kamera"
export DATA_MOUNT_POINT=$(redis-cli --raw -h ${REDIS_HOST} get /sys/arch/base)
export CAM_FOV=$(redis-cli --raw -h ${REDIS_HOST} get /sys/arch/hosts/${NODE_HOSTNAME}/fov)

export ROS_DISTRO="noetic"
export KAMERA_DNS_IP="192.168.88.1"
export PROJ_DIR="${KAMERA_DIR}"
export PULSE_TTY=/dev/ttyS0
export MCC_DAQ="/dev/$(readlink /dev/mcc_daq)"
