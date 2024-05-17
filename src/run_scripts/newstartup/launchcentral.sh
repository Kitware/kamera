#!/usr/bin/env bash

# Bring up central nodes - e.g. INS and DAQ
export COMPOSE_IGNORE_ORPHANS=True

if [[ -z "$@" ]] ; then
    ARGS=(up -d)
else
    ARGS=("$@")
fi

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
cd ${KAM_REPO_DIR}
set -a # automatically export all variables
. ${KAM_REPO_DIR}/.env

source ${KAM_REPO_DIR}/src/run_scripts/inpath/locate_daq

docker compose -f "${KAM_REPO_DIR}/compose/central.yml" "${ARGS[@]}" &
STAT_CENTRAL=$!
set +a

wait $STAT_CENTRAL
