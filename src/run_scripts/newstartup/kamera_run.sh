#!/usr/bin/env bash

## This script provides turnkey operation to spin up the whole system
export COMPOSE_IGNORE_ORPHANS=True # make compose quieter
KAM_REPO_DIR=$(/home/user/.config/kamera/repo_dir.bash)
echo $KAM_REPO_DIR
SYSTEM_NAME=$(cat /home/user/kw/SYSTEM_NAME)
source "${KAM_REPO_DIR}/tmux/${SYSTEM_NAME}/env.sh"

errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

blueprintf() {
    (printf "\e[34m$@\e[0m")
}

if [[ -z "$@" ]] ; then
    ARGS=(start)
else
    ARGS=("$@")

fi

MODE=""
for arg in "${ARGS[@]}" ; do
    blueprintf "arg: ${arg}\n"
    if [[ "${arg}" == "-d" || "${arg}" == "--detach" ]] ; then DETACH="-d" ; fi
    if [[ "${arg}" == "--remove-orphans" ]] ; then REM_ORPH="--remove-orphans" ; fi
    if [[ "${arg}" == "up"      ]] ; then MODE=up ; fi
    if [[ "${arg}" == "start"      ]] ; then MODE=up ; fi
    if [[ "${arg}" == "kill"    ]] ; then MODE=kill ; fi
    if [[ "${arg}" == "restart" ]] ; then MODE=restart ; fi
    if [[ "${arg}" == "stop" ]] ; then MODE=stop ; fi
    if [[ "${arg}" == "down" ]] ; then errcho "Do not use this script to 'down'"; exit 1 ; fi
    if [[ "${arg}" == "rm" ]] ; then MODE=rm ; fi
done

if [[ -z "${MODE}" ]]; then
    errcho "Missing compose mode (up/down etc)"
    exit 1
fi

## === === === === === ===   Env setup  === === === === === === ===
blueprintf "Configuring main KAMERA entrypoint."
# Add detector ENV variables to Redis
source ${KAM_REPO_DIR}/src/cfg/set_detector_read_state.sh
blueprintf "."

MASTER_HOST=$(cq '.master_host')

for VNAME in MASTER_HOST KAM_REPO_DIR
do
  if [[ -z "${!VNAME}" || "${!VNAME}" == 'null' ]]; then
    printf "<kamera_run.sh!> Unable to determine $VNAME: Check user-config\n"
    exit 1
  fi
done

## === === === === === ===   End Env setup  === === === === === ===

## === === === === === Handle drive mounts === === === === ===
echo "Waiting for master host to come online. Hit ctrl-c or window X to cancel"
i=1
SP="/-\|"
until ping -c1 -W1 ${MASTER_HOST} &>/dev/null; do
    printf "\b${SP:i++%${#SP}:1}"
done

SKIP_PING=true
for host in $(cq '.arch.hosts | keys | join("\n" )') ; do
  hostip=$(dig +short $host)
  if [[ $? != 0 ]]; then
    errcho "FATAL: Cannot resolve IP for necassary host ${host}"
  fi

  _CLIENTS=$(redis-cli -h $REDIS_HOST client list | grep $hostip)
  if [[ $? != 0 ]]; then
    SKIP_PING=
  fi
done

if [[ -n ${SKIP_PING} ]]; then
  echo "all clients located, yay!"
else
  for host in $(cq '.arch.hosts | keys | join("\n" )') ; do
      echo "Waiting on ping $host."
      if [[ $(cq ".arch.hosts.${host}.enabled") == 'true' ]]; then
          until ping -c1 -W1 ${host} &>/dev/null; do
              printf "\b${SP:i++%${#SP}:1}"
          done
      else
          echo "${host} disabled."
      fi
  done
fi

supervisorctl restart mount_nas
ls /mnt/flight_data/.flight_data_mounted
NAS_CODE=$?
if ! [[ $NAS_CODE == 0 ]]; then
    echo "Failed to connect to NAS! Troubleshoot!"
    sleep 5
    exit
else
    echo "NAS mounted!"
fi
declare -A PIDS
for host in $(cq '.arch.hosts | keys | join("\n" )') ; do
    if [[ $(cq ".arch.hosts.${host}.enabled") == 'true' ]]; then
	python3 ${KAM_REPO_DIR}/scripts/system.py $host restart nas &
        PIDS[${host}]=$!
    else
        echo "${host} disabled."

    fi
done

for pid in ${PIDS[*]}; do
    wait $pid
done


# Bring up master and core nodes
MASTER_HOST=$(cq '.master_host')
blueprintf "done\nBringing up master $MASTER_HOST..."
python3 ${KAM_REPO_DIR}/scripts/system.py $MASTER_HOST "${ARGS[@]}" master

# check that master is in fact up
FAIL_COUNT=0
cd $KAM_REPO_DIR
until docker compose -f ${KAM_REPO_DIR}/compose/nodelist.yml run --rm nodelist; do
	echo "Attempt $((++FAIL_COUNT))";
	if [[ $FAIL_COUNT -gt 3 ]]; then
	    errcho "Unable to contact ros master. Running WTF and aborting startup"
	    docker compose -f ${KAM_REPO_DIR}/compose/nodelist.yml run --rm nodelist /entry/wat.sh
	    exit 1
	fi
done

# === === === === Checks have passed === === === ===
blueprintf "done. Init checks are good! \nBringing up central..."
python3 ${KAM_REPO_DIR}/scripts/system.py $MASTER_HOST "${ARGS[@]}" central &
STAT_CENTRAL=$!


blueprintf "done\nLaunching pod nodes...\n"

# Bring up all pod systems
# Query list of hosts as line delim array
declare -A PIDS
for host in $(cq '.arch.hosts | keys | join("\n" )') ; do
    if [[ $(cq ".arch.hosts.${host}.enabled") == 'true' ]]; then
	python3 ${KAM_REPO_DIR}/scripts/system.py $host "${ARGS[@]}" pod &
        PIDS[${host}]=$!
    else
        echo "${host} disabled."

    fi
done
echo " PIDS : ${PIDS[@]}"

# === === === === Checks have passed === === === ===
blueprintf "Bringing up monitor..."
python3 ${KAM_REPO_DIR}/scripts/system.py $MASTER_HOST "${ARGS[@]}" monitor &
STAT_MONITOR=$!

blueprintf "done. \nPods launched. Starting GUI..."

cd ${KAM_REPO_DIR}
set -a # automatically export all variables
xhost +local:root
docker compose -f "${KAM_REPO_DIR}/compose/gui.yml" "${ARGS[@]}" &
STAT_GUI=$!
blueprintf "done\n  === Starting System Control Panel :D === "
set +a

gui_splash() {
    notify-send -t 5000 -i ~/kw/kamera/src/cfg/seal-icon.png \
        "KAMERA" "Starting KAMERA Control Panel, please wait" || true
}
gui_splash

wait $STAT_CENTRAL
wait $STAT_MONITOR
# wait for all pids
for pid in ${PIDS[*]}; do
    echo "waiting on: $pid"
    wait $pid
done
wait $STAT_GUI

