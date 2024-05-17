#!/usr/bin/env bash

# Launch all pod nodes (replicated across all systems)
export COMPOSE_IGNORE_ORPHANS=True

if [[ -z "$@" ]] ; then
    ARGS=(up -d)
else
    ARGS=("$@")
fi


#for value in "${ARGS[@]}" ; do    #print the new array
#echo "$value"
#done

KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
cd ${KAM_REPO_DIR}
set -a # automatically export all variables
. ${KAM_REPO_DIR}/.env

# get configQuery
source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh

# infer position from the config based on acting hostname (defaults to actual hostname)
NODE_HOSTNAME=${NODE_HOSTNAME:-$(hostname)}
CAM_FOV=$(cq ".hosts.${NODE_HOSTNAME}.fov")
export NODE_HOSTNAME
export CAM_FOV

for VNAME in CFG_ALIAS_SET NODE_HOSTNAME CAM_FOV
do
  if [[ -z "${!VNAME}" || "${!VNAME}" == 'null' ]]; then
    printf "<launchcam.sh!> Unable to determine $VNAME: ${CAM_FOV}. Check user-config\n"
    exit 1
  fi
done

# crazy script to ensure that ethernet speed is max
# ONLY RUN ONCE AT STARTUP
# set_eth_speed_max.sh


docker compose -f ${KAM_REPO_DIR}/compose/cam.yml "${ARGS[@]}" &
CAM_STAT=$!
set +a

wait $CAM_STAT
docker ps
