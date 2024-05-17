#!/bin/bash
# A bug here: https://bugs.launchpad.net/ubuntu/+source/docker/+bug/1858441
# Details that docker doesn't start on boot, workaround here
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/../env.sh

sudo service docker restart
