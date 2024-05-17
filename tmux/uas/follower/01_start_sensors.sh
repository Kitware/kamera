#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Kills running containers and processes
trap "{ echo Stopping session 'sensors'; tmuxinator stop sensors; exit 0; }" EXIT

source $DIR/../env.sh

echo "Starting session 'sensors'."
ln -sf $DIR/config/sensors.yml ~/.tmuxinator/sensors.yml

tmuxinator start -n sensors -p $DIR/config/sensors.yml

# Keep process alive for supervisor
sleep infinity
