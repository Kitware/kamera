#!/usr/bin/env bash

# check if the nas is available, and try to automount if it is
NAS_HOST=kamera_nas
if ping -c1 -W1 ${NAS_HOST}; then
    mount -a
    NAS_POINT=$(grep "^${NAS_HOST}" /proc/mounts | awk '{print $2}')
    echo "${NAS_POINT}"
    notify-send -t 5000 -i folder-open \
        "NAS mounted" "Mounted NAS to  ${NAS_POINT}" 2>/dev/null || true
    true
else
    notify-send  -t 5000 --urgency=critical -i dialog-warning \
        "NAS mount failed" "Failed to ping NAS host ${NAS_HOST}" || true
    echo "Failed to ping NAS at ${NAS_HOST}"
    false
fi
