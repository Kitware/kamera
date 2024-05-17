#!/bin/bash

# INS node startup script

echo "( ) ( ) ( ) PUBLISH SYNC MSGS ( ) ( ) ( ) "
source /entry/project.sh
source /aliases.sh

# Launch image directory publisher from specified dir
exec roslaunch --wait sprokit_adapters publish_sync_msgs.launch \
    publish_rate:=1 \
    out_topic:="/${NODE_HOSTNAME}/synched" \
    flight_dir:="/mnt/data/testset"
