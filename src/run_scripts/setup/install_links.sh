#!/usr/bin/env bash

## Install breadcrumb links so we can find this repo
HEREDIR=$(cd "$(dirname $(realpath ${0}))" && pwd)
REPO_DIR=$(cd ${HEREDIR}/../../../ && pwd)
echo "REPO_DIR: $REPO_DIR"

if [[ -n `grep docker /proc/1/cgroup` ]] ; then
    export IS_CONTAINER=true
fi

mkdir -p ${HOME}/.config/kamera/
mkdir -p ${HOME}/.local/bin/

ln -sfv ${REPO_DIR}/repo_dir.bash ${HOME}/.config/kamera/repo_dir.bash
ln -sfv ${REPO_DIR}/src/run_scripts/bootstrap_app.sh ${HOME}/.config/kamera/bootstrap_app.sh
ln -sfv ${REPO_DIR}/src/run_scripts/newstartup/kamera_run.sh ${HOME}/.local/bin/kamera_run


if [[ -n $IS_CONTAINER ]]; then
    echo "ln -srf ${REPO_DIR}/src/cfg /"
    ln -srf "${REPO_DIR}/src/cfg /"
fi