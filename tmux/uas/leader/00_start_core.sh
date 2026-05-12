#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cleanup() {
    echo "Stopping session 'core'."
    docker-compose -f "${KAMERA_DIR}/compose/core.yml" down
    tmux kill-session -t core 2>/dev/null
}
trap cleanup EXIT

. "${DIR}/../startup.sh"
export KAMERA_DIR="/home/user/kw/kamera"

echo "Start session 'core'."
tmux new-session -d -s core -n core -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/core.yml up roscore'"
tmux new-window -t core: -n webvideo -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/core.yml up webvideo'"
tmux new-window -t core: -n influxdb -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/core.yml up influxdb'"
tmux new-window -t core: -n diagnostics -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/core.yml up diagnostics2influxdb'"
tmux new-window -t core: -n diagnostic_aggregator -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/core.yml up diagnostic_aggregator'"
tmux new-window -t core: -n cam_param_monitor -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/core.yml up cam_param_monitor'"

sleep infinity
