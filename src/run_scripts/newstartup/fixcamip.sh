#!/usr/bin/env bash

export COMPOSE_IGNORE_ORPHANS=True


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

docker compose -f compose/debug.yml run --rm dbg_cam /entry/make_cameras_ip_static.sh

set +a


