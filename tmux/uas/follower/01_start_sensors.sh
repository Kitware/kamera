#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cleanup() {
    echo "Stopping session 'sensors'."
    docker-compose -f "${KAMERA_DIR}/compose/sensors.yml" down
    tmux kill-session -t sensors 2>/dev/null
}
trap cleanup EXIT

. "${DIR}/../startup.sh"
export KAMERA_DIR="/home/user/kw/kamera"

echo "Starting session 'sensors'."
tmux new-session -d -s sensors -n rgb_cam -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/sensors.yml up cam_rgb'"
tmux new-window -t sensors: -n ir_cam -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/sensors.yml up cam_ir'"

sleep infinity
