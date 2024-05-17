#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/../env.sh

python /home/user/kw/postflight_scripts/scripts/create_flight_summary_redis.py
