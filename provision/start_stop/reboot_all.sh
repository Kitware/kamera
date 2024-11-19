#!/bin/bash

# Location of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

SYSTEM_NAME=$(cat /home/user/kw/SYSTEM_NAME)

cd ${DIR}/../ansible/
source ${DIR}/../ansible/functions.sh
kamera_reboot SYSTEM_NAME
