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
tmux new-session -d -s core -n webvideo -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker-compose -f compose/core.yml up webvideo'"

sleep infinity
