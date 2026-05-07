#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ $# != 1 ]; then
    echo "Must enter session name." >&2
    exit 1
fi
export SESSION="$1"

cleanup() {
    echo "Stopping session ${SESSION}"
    docker compose -f "${KAMERA_DIR}/compose/${SESSION}.yml" down
    tmux kill-session -t "${SESSION}" 2>/dev/null
}
trap cleanup EXIT

. "${DIR}/../startup.sh"
export KAMERA_DIR=$(${HOME}/.config/kamera/repo_dir.bash)

echo "Starting session '${SESSION}'."
tmux new-session -d -s "${SESSION}" -c "${KAMERA_DIR}" \
    "bash -c 'source ${DIR}/../env.sh && docker compose -f compose/${SESSION}.yml up ${SESSION}'"

sleep infinity
