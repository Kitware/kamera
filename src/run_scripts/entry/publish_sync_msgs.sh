#!/bin/bash

# INS node startup script

echo "( ) ( ) ( ) PUBLISH SYNC MSGS ( ) ( ) ( ) "
source /entry/project.sh
source /aliases.sh

# Launch image directory publisher from specified dir
exec ros2 run sprokit_adapters publish_sync_msgs.py --ros-args \
    -p publish_rate:=1.0 \
    -p out_topic:="/${NODE_HOSTNAME}/synched" \
    -p flight_dir:="/mnt/data/testset"
