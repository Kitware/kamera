#!/usr/bin/env bash
# reboot INS
set -euo pipefail

INS_IP="192.168.88.99"
INS_USER="admin"
INS_PW="sudoseals"

curl -u $INS_USER:$INS_PW "http://${INS_IP}/cgi-bin/resetPage.xml?doReset=1"

echo "Shutting down INS..."
sleep 10

echo "Waiting for INS to come online. Hit ctrl-c or window X to cancel"
i=1
SP="/-\|"
until ping -c1 -W1 ${INS_IP} &>/dev/null; do
    printf "\b${SP:i++%${#SP}:1}"
done
echo "Connected!"
sleep 2
