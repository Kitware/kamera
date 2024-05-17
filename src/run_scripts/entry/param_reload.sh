#!/bin/bash

# Core node startup script

echo "[ ] [ ] [ ] Reload Params [ ] [ ] [ ] "

source /entry/project_env.sh

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

rosparam load /cfg/user-config.yml /cfg
