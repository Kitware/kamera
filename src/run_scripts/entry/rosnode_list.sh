#!/bin/bash

# Diagnostics 

echo "[ ] [ ] [ ] ROSNODE CHECK [ ] [ ] [ ] "

source /entry/project_env.sh


# Expected exit code from a Ctrl-C when in explicit docker run mode.
trap "errcho 'Caught SIGINT'; cleanup" SIGINT
# Expected exit code from docker stop command.
trap "errcho 'Caught SIGTERM'; cleanup" SIGTERM

exec rosnode list

