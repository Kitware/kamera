#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ $# != 1 ]
then
  echo "Must enter session name."
  exit 1
fi
export SESSION="$1"

# Kills running containers and processes
trap "{ echo Stopping session ${SESSION}; tmuxinator stop ${SESSION}; exit 0; }" EXIT

source $DIR/../env.sh

echo "Starting session '${SESSION}'."
ln -sf $DIR/config/generic.yml ~/.tmuxinator/${SESSION}.yml

tmuxinator start -n ${SESSION} -p ~/.tmuxinator/${SESSION}.yml

# Keep process alive for supervisor
sleep infinity
