#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Kills running containers and processes
trap "{ echo Stopping session 'processing'; tmuxinator stop processing; exit 0; }" EXIT

source $DIR/../env.sh

echo "Starting session 'processing'."
ln -sf $DIR/config/processing.yml ~/.tmuxinator/processing.yml

tmuxinator start -n processing -p $DIR/config/processing.yml

# Keep process alive for supervisor
sleep infinity
