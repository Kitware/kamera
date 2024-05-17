#!/usr/bin/env bash

# Start the master and center
KAM_REPO_DIR=$(~/.config/kamera/repo_dir.bash)
if [[ -z "${KAM_REPO_DIR}" ]]; then
    echo "ERROR: Could not resolve KAM_REPO_DIR. Check ~/.config/kamera"
    exit 1
fi

MCC_DAQ=`readlink -f /dev/mcc_daq`
export MCC_DAQ
printf "MCC_DAQ.........: ${MCC_DAQ}
"
docker compose -f "${KAM_REPO_DIR}/compose/central.yml" up -d
