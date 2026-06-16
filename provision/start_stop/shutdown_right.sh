#!/bin/bash

# Location of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/../ansible/
source ${DIR}/../ansible/functions.sh
kamera_shutdown $(kamera_host_for_position right)
