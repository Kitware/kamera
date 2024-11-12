#!/usr/bin/env bash

## Like kamera_run, but opposite
export COMPOSE_IGNORE_ORPHANS=True # make compose quiter
source ${KAM_REPO_DIR}/tmux/cas/env.sh


errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

blueprintf() {
    (printf "\e[34m$@\e[0m")
}

if [[ -z "$@" ]] ; then
    ARGS=(stop)
else
    ARGS=("$@")

fi

MODE=""
for arg in "${ARGS[@]}" ; do
    blueprintf "arg: ${arg}\n"
    if [[ "${arg}" == "-d" || "${arg}" == "--detach" ]] ; then DETACH="-d" ; fi
    if [[ "${arg}" == "--remove-orphans" ]] ; then REM_ORPH="--remove-orphans" ; fi
    if [[ "${arg}" == "up"      ]] ; then errcho "Do not use this script to start"; exit 1; fi
    if [[ "${arg}" == "kill"    ]] ; then MODE=kill ; fi
    if [[ "${arg}" == "restart" ]] ; then MODE=restart ; fi
    if [[ "${arg}" == "stop" ]] ; then MODE=stop ; fi
    if [[ "${arg}" == "down" ]] ; then MODE=down ; fi
    if [[ "${arg}" == "rm" ]] ; then MODE=rm ; fi
done

if [[ -z "${MODE}" ]]; then
    errcho "Missing compose mode (up/down etc)"
    exit 1
fi

## === === === === === ===   Env setup  === === === === === === ===
blueprintf "Configuring main KAMERA entrypoint."
source ~/.config/kamera/repo_dir.bash
if [[ -z "${KAM_REPO_DIR}" ]]; then
    echo "ERROR: Could not resolve KAM_REPO_DIR. Check ~/.config/kamera"
    exit 1
fi
blueprintf "."

# make our utility scripts runnable
PATH="${KAM_REPO_DIR}/src/run_scripts/inpath:$PATH"
export PATH
blueprintf "."


# load cq - ConfigQuery
source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh
blueprintf "."

STARTDIR=${KAM_REPO_DIR}/src/run_scripts/newstartup
blueprintf ".done\n"

MASTER_HOST=$(cq '.master_host')
## === === === === === ===   End Env setup  === === === === === ===


OPTIONS=(${VERBOSE})

blueprintf "done\nBringing down gui..."
PREFIX_DC_BASH=xhost_disable wrap-docker-compose -f "${KAM_REPO_DIR}/compose/gui.yml" "${ARGS[@]}" &
STAT_GUI=$!

blueprintf "done\nGui down. Killing pods...\n"
# Bring up all pod systems
# Query list of hosts as line delim array
declare -A PIDS
for host in $(cq '.arch.hosts | keys | join("\n" )') ; do
    if [[ $(cq ".arch.hosts.${host}.enabled") == 'true' ]]; then
	python3 ${KAM_REPO_DIR}/scripts/system.py $host ${ARGS[@]} pod &
        PIDS["${host}_pod"]=$!
	python3 ${KAM_REPO_DIR}/scripts/system.py $host ${ARGS[@]} detector &
        PIDS["${host}_det"]=$!
    fi
done
blueprintf "done\nBringing down central..."
python3 ${KAM_REPO_DIR}/scripts/system.py $MASTER_HOST ${ARGS[@]} central &
PIDS["${host}_cen"]=$!
python3 ${KAM_REPO_DIR}/scripts/system.py $MASTER_HOST ${ARGS[@]} monitor &
PIDS["${host}_mon"]=$!
wait $STAT_GUI
# wait for all pids
for pid in ${PIDS[*]}; do
    wait $pid
done

blueprintf "done\nBringing down master..."
python3 ${KAM_REPO_DIR}/scripts/system.py $MASTER_HOST ${ARGS[@]} master
blueprintf "done. ROS should be down\n"

blueprintf "Dismounting drives..."
# todo: dismounts
blueprintf "done\n "
docker ps
