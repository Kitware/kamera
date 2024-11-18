#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

xhost +

source $DIR/startup.sh
source $DIR/env.sh

if [ ${REDIS_HOST} = "localhost" ]; then
    roscore&
fi

echo "Start gui."
cd $DIR/../..

docker compose -f $KAMERA_DIR/compose/gui.yml up
