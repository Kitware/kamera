#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)

cd ${KAM_REPO_DIR}
# get configQuery
source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh

set -a # automatically export all variables

. ${KAM_REPO_DIR}/.env
REDIS_HOST="192.168.88.10"
REDIS_PORT="6379"

UP_RESET="redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} set /${host}/detector/up False"
DOWN_RESET="redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} set /${host}/detector/down False"

declare -A PREV

while true; do
    declare -A PIDS

    for host in $(cq '.hosts | keys | join("\n" )') ; do
        PIPEFILE=$(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} get "/${host}/detector/pipefile")
        START_DETECTOR=$(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} get "/${host}/detector/up")
        STOP_DETECTOR=$(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} get "/${host}/detector/down")
    
        if [ "${PREV["${host}-pipe"]}" != "$PIPEFILE" ]
        then
            echo "==============================================================="
            echo "Loading pipefile for host $host: $PIPEFILE"
            PREV["${host}-pipe"]=${PIPEFILE}
            ${KAM_REPO_DIR}/src/run_scripts/inpath/ssh_checked.sh $host \
                "${KAM_REPO_DIR}/src/run_scripts/inpath/kamera.detector restart" &
            PIDS["${host}-pipe"]=$!
        fi
 
        if [ "${PREV["${host}-start"]}" != "$START_DETECTOR" ]
        then
            if [ "${START_DETECTOR}" == "True" ]
            then
                echo "==============================================================="
                echo "Starting Detector: $host"
                PREV["${host}-start"]=${START_DETECTOR}
                ${KAM_REPO_DIR}/src/run_scripts/inpath/ssh_checked.sh $host \
                    "${KAM_REPO_DIR}/src/run_scripts/inpath/kamera.detector" &
                PIDS["${host}-start"]=$!
                # Reset redis parameter, so each "set" triggers a call
                RET=$(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} set "/${host}/detector/up" False)
            else
                PREV["${host}-start"]=${START_DETECTOR}
            fi
        fi
 
        if [ "${PREV["${host}-stop"]}" != "$STOP_DETECTOR" ]
        then
            if [ "$STOP_DETECTOR" == "True" ]
            then
                echo "==============================================================="
                echo "Stopping Detector: $host"
                PREV["${host}-stop"]=${STOP_DETECTOR}
                ${KAM_REPO_DIR}/src/run_scripts/inpath/ssh_checked.sh $host \
                    "${KAM_REPO_DIR}/src/run_scripts/inpath/kamera.detector down" &
                PIDS["${host}-stop"]=$!
                RET=$(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} set "/${host}/detector/down" False)
            else
                PREV["${host}-stop"]=${STOP_DETECTOR}
            fi
        fi
    done

    for pid in ${PIDS[*]}; do
        wait $pid
        echo "Finished SSH call."
    done
    unset PIDS
    sleep 0.1
done
