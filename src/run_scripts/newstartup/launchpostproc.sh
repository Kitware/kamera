#!/usr/bin/env bash

export COMPOSE_IGNORE_ORPHANS=True

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
cd ${KAM_REPO_DIR}

set -a # automatically export all variables

. ${KAM_REPO_DIR}/.env

RUNSCRIPT=$1
export KAM_FLIGHT_DIR="$2"
echo docker compose -f "${KAM_REPO_DIR}/compose/postproc.yml" run --rm postproc $RUNSCRIPT
docker compose -f "${KAM_REPO_DIR}/compose/postproc.yml" run --rm postproc $RUNSCRIPT

set +a

sleep 1

