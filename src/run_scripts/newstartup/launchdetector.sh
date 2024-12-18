#!/usr/bin/env bash

export COMPOSE_IGNORE_ORPHANS=True

if [[ -z "$@" ]] ; then
    ARGS=(up -d)
else
    ARGS=("$@")
fi

gui_splash() {
   MSG_STRING="${1}ing KAMERA Detector, please wait"
    notify-send -t 5000 -i ~/kw/kamera/src/cfg/seal-icon.png \
        "KAMERA" "${MSG_STRING}" 2>/dev/null || echo "${MSG_STRING}";
}

for arg in "${ARGS[@]}" ; do
    if [[ "${arg}" == "up"      ]] ; then gui_splash 'start' ; fi
    if [[ "${arg}" == "restart" ]] ; then gui_splash 'restart' ; fi
done

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
cd ${KAM_REPO_DIR}
# get configQuery
source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh

set -a # automatically export all variables

. ${KAM_REPO_DIR}/.env



# infer position from the config based on acting hostname (defaults to actual hostname)
NODE_HOSTNAME=${NODE_HOSTNAME:-$(hostname)}
CAM_FOV=$(cq ".hosts.${NODE_HOSTNAME}.fov")
export NODE_HOSTNAME
export CAM_FOV

docker compose -f "${KAM_REPO_DIR}/compose/detector.yml" "${ARGS[@]}"
STAT_DETECTOR=$!

set +a

wait $STAT_GUI
sleep 1

