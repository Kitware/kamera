#!/bin/bash


source_dir='/mnt/data/'
dest_dir="/mnt/flight_data/"

RESP=$(redis-cli -h ${REDIS_HOST} ping)
while [ "$RESP" != "PONG" ]
do
    echo "Got '$RESP', wanted 'PONG'."
    echo "Waiting for redis host $REDIS_HOST to come online..."
    RESP=$(redis-cli -h ${REDIS_HOST} ping)
    sleep 1;
done
echo "Redis successfully connected at $REDIS_HOST, starting."

ls /mnt/flight_data/.flight_data_mounted
NAS_CODE=$?
if ! [[ $NAS_CODE == 0 ]]; then
    echo "Failed to connect to NAS! Troubleshoot!"
    sleep 1
    exit 1
else
    echo "NAS mounted!"
fi

while [ true ]
do
    # STAGE 1
    echo "Gathering and copying files under $source_dir"
    # Find all files that are appended to and copy over incrementally.
    # This is detection files, log files, and ins dat files
    fdfind . $source_dir --exclude '*.{IIQ,iiq,json,tif,jpg}' --type f --exec echo {} | \
        # remove far left path (/mnt/data/) dir for rsync
        cut -d'/' -f4- | \
        # cap max argument length for rsync
        head -n 1000 | \
        # copy all files removing source files on success
        rsync -a --recursive --files-from=- --chmod=777 $source_dir $dest_dir
    echo "Finished copying."

    # STAGE 2
    echo "Copying & removing data files older than 5 minutes in $source_dir"
    # Find all files modified before N, excluding IIQ images, that are not append to.
    # This includes things like tif, jpg, json.
    fdfind . $source_dir --exclude '*.{IIQ,iiq,txt,csv,dat}' --changed-before 5min --type f --exec echo {} | \
        # remove far left dir for rsync
        cut -d'/' -f4- | \
        # cap max argument length for rsync
        head -n 1000 | \
        # copy all files removing source files on success
        rsync -a --recursive --remove-source-files --files-from=- --chmod=777 $source_dir $dest_dir
    echo "Finished copying and removing image files."

    # STAGE 3
    echo "Copying & removing text files older than 1 day in $source_dir"
    ## Find all files that are appended to, which includes csvs, text files, and the .dat files
    ## The one day wait ensures that file is finished being written to.
    fdfind . $source_dir --exclude '*.{IIQ,iiq,json,tif,jpg}' --changed-before 1day --type f --exec echo {} | \
        # remove far left dir for rsync
        cut -d'/' -f4- | \
        # cap max argument length for rsync
        head -n 1000 | \
        # copy all files removing source files on success
        rsync -a --recursive --remove-source-files --files-from=- --chmod=777 $source_dir $dest_dir
    echo "Finished removing text files."

    sleep 1
done
