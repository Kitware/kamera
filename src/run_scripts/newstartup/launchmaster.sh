#!/usr/bin/env bash

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

docker compose -f ${KAM_REPO_DIR}/compose/master.yml "${ARGS[@]}"
set +a