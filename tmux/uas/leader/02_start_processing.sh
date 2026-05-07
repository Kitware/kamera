#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cleanup() {
    echo "Stopping session 'processing'."
    docker-compose -f "${KAMERA_DIR}/compose/processing.yml" down
    tmux kill-session -t processing 2>/dev/null
}
trap cleanup EXIT

. "${DIR}/../startup.sh"
export KAMERA_DIR="/home/user/kw/kamera"

echo "Starting session 'processing'."
tmux new-session -d -s processing -n viewport_rgb -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/processing.yml up viewport_rgb'"
tmux new-window -t processing: -n viewport_ir -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/processing.yml up viewport_ir'"
tmux new-window -t processing: -n nexus -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/processing.yml up nexus'"
tmux new-window -t processing: -n shapefile_monitor -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/processing.yml up shapefile_monitor'"

sleep infinity
