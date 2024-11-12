#!/bin/bash

# INS node startup script

echo "( ) ( ) ( ) INS ( ) ( ) ( ) "
source /entry/project.sh
source /aliases.sh

# INS really ought to be run with kamcore and hence should not use spoof here
if [[ -n ${SPOOF_INS} ]] ; then
    echo "spoof mode"
    ARG_SPOOF="spoof:=${SPOOF_INS}"
fi
exec roslaunch --wait ins_driver ins.launch ${ARG_SPOOF} norespawn:="${NORESPAWN}"
