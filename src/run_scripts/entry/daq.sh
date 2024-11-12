#!/bin/bash

# DAQ node startup script

echo "<=> <=> <=>  DAQ  <=> <=> <=> "
source /entry/project.sh
source /aliases.sh
SPOOF_INS=${SPOOF_INS:-0}

if [[ $(redis-cli --raw -h $REDIS_HOST get /debug/enable ) == "true" ]]; then
  export KAMERA_DEBUG=true
fi

if [[ $(redis-cli --raw -h $REDIS_HOST get /debug/rebuild ) == "true" ]]; then
  echo "/debug/rebuild set, triggering rebuild on startup"
  catkin build mcc_daq
  if [[ $? -ne 0 ]]; then
    echo "Rebuild failed. Your code is in an unstable state"
    exit 1
  else
    echo "Rebuild succeeded"
  fi
fi

if [[ "$MCC_DAQ" == *"tty"* ]] ; then
    export DAQ_TTY="$MCC_DAQ"
    exec roslaunch --wait ser_daq ser_daq.launch norespawn:=${NORESPAWN}"
else
    exec roslaunch --wait mcc_daq daq.launch norespawn:="${NORESPAWN}"
fi
