#!/bin/bash

# Core node startup script

echo "[ ] [ ] [ ] KAMCORE [ ] [ ] [ ] "
# dump the global config as a debugging step
cat /cfg/${SYSTEM_NAME}/config.yaml

source /entry/project_env.sh


function cleanup {
  pkill -2 kamcore
}

# Expected exit code from a Ctrl-C when in explicit docker run mode.
trap "errcho 'Caught SIGINT'; cleanup" SIGINT
# Expected exit code from docker stop command.
trap "errcho 'Caught SIGTERM'; cleanup" SIGTERM

ping -c1 kameramaster

# this is a rolling counter just for fun, and also serves as something any client can always grab
REDIS_HOST=${REDIS_HOST:-nuvo0}
redis-client -h ${REDIS_HOST} incr term
# Start core and block until it's up, then bootstrap parameters
roscore &

# check that master is in fact up
FAIL_COUNT=0
until /entry/rosnode_list.sh; do
    sleep 1
	echo "Attempt $((++FAIL_COUNT))";
	if [[ $FAIL_COUNT -gt 3 ]]; then
	    errcho "Unable to contact ros master. Running WTF and aborting startup"
	    /entry/wat.sh
	    exit 1
	fi
done

rosparam load /cfg/${SYSTEM_NAME}/config.yaml /cfg
exec roslaunch kamcore kamcore.launch data_mount_point:=$DATA_MOUNT_POINT \
    spoof_rate:="${SPOOF_RATE}"
