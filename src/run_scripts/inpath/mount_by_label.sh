#!/usr/bin/env bash

source ~/.bash_aliases

show_help() {
    printf "Usage: mount_by_label LABEL DESTPATH [-u]
    Mount or unmount a drive based on label LABEL to destination DESTPATH
    Options:
        -u, --umount        unmount the drive

    ¯\_(ツ)_/¯\n"
        exit 0

}

locate_by_label() {
    devid=$(sudo blkid | grep -P "(?<=LABEL\=\")($1)+\"" | grep -Po "(\/dev\/sd\w\d)")
    if [[ -z $devid ]] ; then
        errcho "Unable to find device with label $1"
        exit 1
    else
        printf "$devid"
    fi

}


if [[ $1 == "help" || $1 == "--help" || $1 == "-h" ]] ; then
    show_help
fi

## mount a device by it's label
if [[ -z $1 ]] ; then
    errcho "Must provide a label to search for"
    exit 2
else
    label=$1
fi

if [[ $2 == "-u" || $2 == "--umount" || $2 == "--unmount" ]] ; then
    do_unmount=true
    echo "unmounting"
fi

if [[ -z $2 && -z $do_unmount ]] ; then
    errcho "Must provide a destination mount point"
    exit 3
else
    destpath=$2
fi

devid=$(locate_by_label $label)

if [[ -n $do_unmount ]] ; then
    ERROR="$(sudo umount $devid 2>&1 >/dev/null)"
    if [[ -n $ERROR ]] ; then
        errcho $ERROR
        exit 4
    else
        echo "unmounted $label at $devid"
    fi
else
    sudo mkdir -p $destpath
    ERROR="$(sudo mount $devid $destpath 2>&1 >/dev/null)"
    if [[ -n $ERROR ]] ; then
        errcho $ERROR
        exit 5
    else
        echo "Success! mounted label=$label  devid=$devid to $destpath"
    fi
fi


