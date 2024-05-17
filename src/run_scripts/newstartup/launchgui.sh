#!/usr/bin/env bash

export COMPOSE_IGNORE_ORPHANS=True

if [[ -z "$@" ]] ; then
    ARGS=(up -d)
else
    ARGS=("$@")
fi

gui_splash() {
    notify-send -t 5000 -i ~/kw/noaa_kamera/src/cfg/seal-icon.png \
        "KAMERA" "Starting KAMERA Control Panel, please wait" || true
}

for arg in "${ARGS[@]}" ; do
    if [[ "${arg}" == "up"      ]] ; then xhost +local:root; gui_splash ; fi
done

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
cd ${KAM_REPO_DIR}
set -a # automatically export all variables
source ${KAM_REPO_DIR}/tmux/mas/env.sh
xhost +local:root
docker-compose -f "${KAM_REPO_DIR}/compose/gui.yml" "${ARGS[@]}"
STAT_GUI=$!

set +a

wait $STAT_GUI

for arg in "${ARGS[@]}" ; do
    if [[ "${arg}" == "kill" ]] ; then xhost_disable ; fi
    if [[ "${arg}" == "stop" ]] ; then xhost_disable ; fi
    if [[ "${arg}" == "down" ]] ; then xhost_disable ; fi
    if [[ "${arg}" == "rm"   ]] ; then xhost_disable ; fi
done
