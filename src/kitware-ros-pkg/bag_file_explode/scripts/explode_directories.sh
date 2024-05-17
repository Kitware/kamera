#!/usr/bin/env bash
#
# Explode more than one bag directory to the output directory
#

yaml_config="${1}"
shift
output_dir="${1}"
shift
# Remaining args are bag file paths.
set -e

if [ -z "${yaml_config}" ]
then
    echo "ERROR: Require YAML configuration filepath as argument 1"
    exit 1
fi

if [ -z "${output_dir}" ]
then
    echo "ERROR: Require output directory path as argument 2"
    exit 1
fi

if [ "$#" -eq 0 ]
then
    echo "ERROR: Require 1 or more bag directories to explode images from."
    exit 1
fi

for D in $@*
do
    for F in $D/*.bag
    do
        echo "Exploding bag file: ${F}"
        rosrun bag_file_explode rosbag_to_png.py \
            "${yaml_config}" \
            "${output_dir}$(basename $D)" \
            "${F}"
    done
done
