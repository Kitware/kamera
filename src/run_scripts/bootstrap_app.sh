#!/usr/bin/env bash

# Bootstrap the application and bring all of the necessary configuration hooks online
# This needs to be linked in to ~/.config/kamera/
# this will initialize cq (configQuery), KAM_REPO_DIR
errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

command -v jq >/dev/null 2>&1 || { errcho "App requires jq but it's not installed.  Aborting."; exit 127; }
command -v yq >/dev/null 2>&1 || { errcho "App requires yq but it's not installed.  Aborting."; exit 127; }

THIS_FILE=${BASH_SOURCE[0]:-${(%):-%N}} # bash/zsh equivalent

if [[ -z ${THIS_FILE} ]]; then
    errcho "File cannot find itself. Only bash/zsh supported (bash preferred)"
fi

RUN_SCRIPTS_DIR=$(dirname $(realpath ${THIS_FILE}))
KAM_REPO_DIR=$(realpath "${RUN_SCRIPTS_DIR}/../..")

if [[ -z "${KAM_REPO_DIR}" ]]; then
    errcho "ERROR: Could not resolve KAM_REPO_DIR. Check ~/.config/kamera"
    exit 1
fi

if [[ ! -f "${KAM_REPO_DIR}/src/run_scripts/bootstrap_app.sh" ]]; then
    errcho "App is having an existential crisis. 'bootstrap_app' cannot find itself. Check env/configs"
    errcho " Dollar-zero: ${0} \n KAM_REPO_DIR: ${KAM_REPO_DIR} \n BASH_SOURCE: ${BASH_SOURCE}"
    exit 1
fi

# === config helper
CFG_DIR="${KAM_REPO_DIR}/src/cfg"
CFG_FILENAME='user-config.yml'
export CFG_ALIAS_SET=true

# ConfigQuery
cq () {
    CFG_FILE=${CFG_DIR}/user-config.yml ${CFG_DIR}/get "$@"
}

TEST_MASTER_HOST=$(cq '.master_host')

if [[ -z $TEST_MASTER_HOST ]] ; then
    errcho "ERROR: Could not resolve config key 'master_host'. Check cq, ~/.config/kamera"
    exit 1

fi
if [[ $TEST_MASTER_HOST == 'null' ]] ; then
    errcho "ERROR: Value of 'master_host' is '$TEST_MASTER_HOST'. \
    \n Check these paths/files: \
    \n ~/.config/kamera \
    \n ${CFG_DIR}/${CFG_FILENAME} \
    \n ${0}"
    exit 1

fi

export KAM_REPO_DIR
