#!/bin/bash

# Diagnostics 

errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

echo "[?] [?] [?]  WAT. [?] [?] [?]  "

source /entry/project_env.sh

MASTER_HOST=$(echo $ROS_MASTER_URI| grep -Po -e '(?<=http:\/\/)([\w\.]+)(?=:)')
echo "MASTER_HOST: "

# Expected exit code from a Ctrl-C when in explicit docker run mode.
trap "errcho 'Caught SIGINT'; cleanup" SIGINT
# Expected exit code from docker stop command.
trap "errcho 'Caught SIGTERM'; cleanup" SIGTERM

echo "=== === === === Looking for ROS_MASTER === === === === "
echo "=== === === === /etc/hosts: === === === === "
cat /etc/hosts

echo "=== === === === /etc/resolv.conf: === === === === "
cat /etc/resolv.conf

echo "=== === === === ping ${MASTER_HOST} === === === === "
ping -c1 -W1 "${MASTER_HOST}"

echo "=== === === === nslookup ${MASTER_HOST} === === === === "
nslookup "${MASTER_HOST}"

echo "=== === === === dig ${MASTER_HOST} === === === === "
dig ${MASTER_HOST}
MASTER_IP=$(dig +short ${MASTER_HOST})
echo $MASTER_IP

if [[ -z $MASTER_IP ]]; then
    errcho "FATAL. Cannot resolve to master IP. This is a nonstarter \n :( :( :("
    exit 1
fi

echo "=== === === === dig MASTER_IP (${MASTER_IP}) === === === === "
nslookup $MASTER_IP

exec roswtf

