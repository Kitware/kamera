#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

xhost +

source $DIR/startup.sh
source $DIR/env.sh

if [ ${REDIS_HOST} = "localhost" ]; then
    roscore&
fi

supervisorctl restart flight_summary

echo "Setting default redis params."
cat $DIR/default_params.conf | xargs -n 2 bash -c 'redis-cli -h ${REDIS_HOST} set $0 $1'

echo "Start gui."
cd $DIR/../..

docker-compose -f $KAMERA_DIR/compose/uas_gui.yml up
