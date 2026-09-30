#!/usr/bin/env bash

## Behind `kamera system {start,stop,restart,status}` and `kamera gui`: bring the
## whole system up or down across every enabled host, through each host's
## supervisor (scripts/system.py), or open the GUI.
export COMPOSE_IGNORE_ORPHANS=True # make compose quieter
KAM_REPO_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
SYSTEM_PY="${KAM_REPO_DIR}/scripts/system.py"
GUI_COMPOSE="${KAM_REPO_DIR}/compose/gui.yml"

errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

blueprintf() {
    (printf "\e[34m$@\e[0m")
}

# Enabled hosts from the config, sorted so hosts are started idempotently.
enabled_hosts() {
    for host in $(jq -r '.arch.hosts | keys | join("\n")' "$KAMERA_CFG" | sort); do
        if [[ $(jq -r ".arch.hosts.${host}.enabled" "$KAMERA_CFG") == 'true' ]]; then
            echo "${host}"
        fi
    done
}

# Exits 0 when the ROS master answers.
ros_master_up() {
    docker compose -f "${KAM_REPO_DIR}/compose/nodelist.yml" run --rm nodelist > /dev/null 2>&1
}

gui_running() {
    [[ -n $(docker compose -f "${GUI_COMPOSE}" ps -q gui 2>/dev/null) ]]
}

do_start() {
    echo "Checking if system is up..."
    if ros_master_up; then
        blueprintf "System already up. Use \`kamera gui\` to open the control panel, or \`kamera system restart\`.\n"
        return 0
    fi

    blueprintf "Configuring main KAMERA entrypoint."
    # Add detector ENV variables to Redis
    source "${KAM_REPO_DIR}/scripts/set_detector_read_state.sh"
    blueprintf ".\n"

    ## === === === === === Handle drive mounts === === === === ===
    echo "Waiting for master host to come online. Hit ctrl-c or window X to cancel"
    local i=1
    local SP="/-\|"
    until ping -c1 -W1 "${MASTER_HOST}" &>/dev/null; do
        printf "\b${SP:i++%${#SP}:1}"
    done

    local SKIP_PING=true
    local host
    for host in $(jq -r '.arch.hosts | keys | join("\n")' "$KAMERA_CFG"); do
        local hostip
        hostip=$(dig +short "${host}")
        if [[ -z "${hostip}" ]]; then
            errcho "Cannot resolve IP for host ${host}"
            SKIP_PING=
        elif ! redis-cli -h "${REDIS_HOST}" client list | grep -q "${hostip}"; then
            SKIP_PING=
        fi
    done

    if [[ -n ${SKIP_PING} ]]; then
        echo "all clients located, yay!"
    else
        for host in $(jq -r '.arch.hosts | keys | join("\n")' "$KAMERA_CFG"); do
            echo "Waiting on ping ${host}."
            if [[ $(jq -r ".arch.hosts.${host}.enabled" "$KAMERA_CFG") == 'true' ]]; then
                until ping -c1 -W1 "${host}" &>/dev/null; do
                    printf "\b${SP:i++%${#SP}:1}"
                done
            else
                echo "${host} disabled."
            fi
        done
    fi

    supervisorctl restart mount_nas
    if ls /mnt/flight_data/.flight_data_mounted; then
        echo "NAS mounted!"
    else
        echo "Failed to connect to NAS! Troubleshoot!"
        sleep 5
    fi
    local pids=()
    for host in $(enabled_hosts); do
        python3 "${SYSTEM_PY}" "${host}" restart nas &
        pids+=($!)
    done
    wait "${pids[@]}"

    # Bring up master and core nodes
    blueprintf "Bringing up master ${MASTER_HOST}..."
    python3 "${SYSTEM_PY}" "${MASTER_HOST}" start master

    # check that master is in fact up
    local FAIL_COUNT=0
    until docker compose -f "${KAM_REPO_DIR}/compose/nodelist.yml" run --rm nodelist; do
        echo "Attempt $((++FAIL_COUNT))"
        if [[ $FAIL_COUNT -gt 3 ]]; then
            errcho "Unable to contact ros master. Running WTF and aborting startup"
            docker compose -f "${KAM_REPO_DIR}/compose/nodelist.yml" run --rm nodelist /entry/wat.sh
            exit 1
        fi
    done

    # === === === === Checks have passed === === === ===
    blueprintf "done. Init checks are good! \nBringing up central..."
    pids=()
    python3 "${SYSTEM_PY}" "${MASTER_HOST}" start central &
    pids+=($!)

    blueprintf "done\nLaunching pod nodes...\n"
    for host in $(enabled_hosts); do
        # daemon group (kamerad) is always started and never stopped by normal operations
        python3 "${SYSTEM_PY}" "${host}" start daemon
        python3 "${SYSTEM_PY}" "${host}" start pod &
        pids+=($!)
    done

    blueprintf "Bringing up monitor..."
    python3 "${SYSTEM_PY}" "${MASTER_HOST}" start monitor &
    pids+=($!)
    wait "${pids[@]}"
    blueprintf "done. System is up; \`kamera gui\` opens the control panel.\n"
}

do_stop() {
    blueprintf "Bringing down gui..."
    docker compose -f "${GUI_COMPOSE}" down &
    local pids=($!)

    blueprintf "done\nStopping pods...\n"
    local host
    for host in $(enabled_hosts); do
        python3 "${SYSTEM_PY}" "${host}" stop pod &
        pids+=($!)
        python3 "${SYSTEM_PY}" "${host}" stop detector &
        pids+=($!)
    done
    blueprintf "done\nBringing down central..."
    python3 "${SYSTEM_PY}" "${MASTER_HOST}" stop central &
    pids+=($!)
    python3 "${SYSTEM_PY}" "${MASTER_HOST}" stop monitor &
    pids+=($!)
    wait "${pids[@]}"

    blueprintf "done\nBringing down master..."
    python3 "${SYSTEM_PY}" "${MASTER_HOST}" stop master
    blueprintf "done. ROS should be down\n"
    docker ps
}

do_status() {
    if ros_master_up; then
        blueprintf "ROS master ${MASTER_HOST}: up\n"
    else
        errcho "ROS master ${MASTER_HOST}: unreachable"
    fi
    if gui_running; then
        blueprintf "GUI: running\n"
    else
        echo "GUI: not running"
    fi
    local host
    for host in $( (echo "${MASTER_HOST}"; enabled_hosts) | sort -u); do
        python3 "${SYSTEM_PY}" "${host}" status
    done
}

# Stays in the foreground until the GUI closes.
do_gui() {
    notify-send -t 5000 "KAMERA" "Starting KAMERA Control Panel, please wait" || true
    xhost +local:root
    exec docker compose -f "${GUI_COMPOSE}" up
}

## === === === === === ===   Env setup  === === === === === === ===
source "${KAM_REPO_DIR}/runtime/env.sh"
# The compose files mount ${PWD}/src.
cd "${KAM_REPO_DIR}"
MASTER_HOST=$(jq -r '.master_host' "$KAMERA_CFG")
if [[ -z "${MASTER_HOST}" || "${MASTER_HOST}" == 'null' ]]; then
    errcho "Unable to determine MASTER_HOST: check src/cfg/${SYSTEM_NAME}/config.json"
    exit 1
fi

case "$1" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        reopen_gui=
        if gui_running; then reopen_gui=true; fi
        do_stop
        do_start
        if [[ -n ${reopen_gui} ]]; then
            do_gui
        fi
        ;;
    status)
        do_status
        ;;
    gui)
        do_gui
        ;;
    *)
        errcho "usage: system.sh {start,stop,restart,status,gui}"
        exit 2
        ;;
esac
