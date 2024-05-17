#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/../env.sh

trap "{sudo service flight_summary stop; exit 0}" EXIT

sudo service flight_summary restart

sleep infinity
