#!/usr/bin/env bash

FAIL_ON_CHECK_COMMAND=${FAIL_ON_CHECK_COMMAND}

logbold() {
  (echo >&2 -e "\e[1m$1\e[0m")
}

logred() {
  (echo >&2 -e "\e[31m$1\e[0m")
}

loggrn() {
  (echo >&2 -e "\e[32m$1\e[0m")
}

logyel() {
  (echo >&2 -e "\e[33m$1\e[0m")
}

logmag() {
  (echo >&2 -e "\e[35m$1\e[0m")
}


function check_command() {
  if ! command -v "${2}" &>/dev/null; then
    if [[ "$1" == "E" ]]; then
      MSG="command could not be found. Please make sure it is installed and in your PATH:\e[1m $2"
      logred "$MSG"
      if [[ -n "${FAIL_ON_CHECK_COMMAND}" ]]; then
        exit 1
      fi
    else
      logyel "MISSING :\e[1m $2"
    fi
  else
    loggrn "FOUND   :\e[1m $2"
  fi
}

function check_run() {
  MSG="${1}"
  ARGS=( "${@:2}" )
  logmag "checking: ${ARGS[*]}"

  if ! ${ARGS[*]} 1>/dev/null; then
    MSG="FAILED  : ${MSG}: ${ARGS[*]}"
    logred "${MSG}"
  else
    loggrn "SUCCESS :\e[1m ${MSG}"
  fi
}


function check_result() {
  MSG="${1}"
  LVL="${2}"
  RES="${3}"

  if [[ -z "${RES}" ]]; then
    if [[ "${LVL}" == "E" ]]; then
      logred "FAILED  :\e[1m ${MSG}: false"
    else
      logyel "WARNING :\e[1m ${MSG}: false"
    fi
  else
    loggrn "SUCCESS :\e[1m ${MSG}"
  fi
}

function check_exists() {
  PATH="${1}"
  LVL="${2}"

  if [[ ! -e "${PATH}" ]]; then
    if [[ "${LVL}" == "E" ]]; then
      logred "MISSING :\e[1m ${PATH}"
    else
      logyel "MISSING :\e[1m ${PATH}"
    fi
  else
    loggrn "FOUND   :\e[1m ${PATH}"
  fi
}


if ! echo hi | grep -P hi &>/dev/null; then
  errcho "Error: You need gnu grep (with -P flag) in your PATH"; exit 1
fi

logbold "checking for which necessary commands are available:"

check_command E docker
check_command E docker compose
check_command E python3
check_command E git
check_command E ifconfig

logbold "checking for commands necessary for Nuvos:"
check_command W nvidia-smi
check_command W redis-cli
check_command W redis-server
check_command W yq
check_command W jq

logbold "checking other conditions:"

for VNAME in REDIS_HOST ROS_MASTER_URI
do
  if [[ -z "${!VNAME}" ]]; then
    logyel "UNSET   :\e[1m ${VNAME}="
  elif [[ "${!VNAME}" == 'null' ]] ; then
    echo "ERROR: $VNAME is null. check config"
  else
    loggrn "INFO    :\e[1m ${VNAME}=${!VNAME}"
  fi
done

check_result "user in docker  group" E "$(groups | grep docker)"
check_result "user in dialout group" W "$(groups | grep dialout)"
check_run "connect to redis" "redis-cli -h ${REDIS_HOST} client list"
check_result "interface pci_top  exists" W "$(ifconfig | grep pci_top)"
check_result "interface pci_btm  exists" W "$(ifconfig | grep pci_btm)"
check_result "interface mobo_top exists" W "$(ifconfig | grep mobo_top)"
check_result "interface mobo_btm exists" W "$(ifconfig | grep mobo_btm)"
check_result "network is on  192.168.88" W "$(ifconfig | grep '192.168.88')"
check_exists "$HOME/kw/noaa_kamera" E

