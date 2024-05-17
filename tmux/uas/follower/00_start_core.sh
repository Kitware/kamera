#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Kills running containers and processes
trap "{ echo Stopping session 'core'.; tmuxinator stop core; exit 0; }" EXIT

source $DIR/../env.sh

echo "Start session 'core'."
ln -sf $DIR/config/core.yml ~/.tmuxinator/core.yml

tmuxinator start -p $DIR/config/core.yml

# Keep process alive for supervisor
sleep infinity
