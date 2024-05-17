#!/bin/bash
# Place this file at /usr/local/bin/resolve_mcc_daq.sh (see 61-mcc.rules for RUN target)

# Autorun script that triggers on hotplug
# right now, it just locates the device mcc_daq in /dev/bus/usb and pushes
# that location to a known spot on the filesystem. Eventually this should
# interface with swarm better.
mkdir -p /var/dev
printf "#!/bin/sh
export MCC_DAQ=$(readlink /dev/mcc_daq)
" > /var/dev/env_mcc_daq.sh
chmod +x /var/dev/env_mcc_daq.sh

