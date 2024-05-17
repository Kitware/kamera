#!/bin/bash
set -e
MP4DIR=${1}
for mp4 in ${MP4DIR}/*.mp4; do
    ./run.sh $mp4
done
