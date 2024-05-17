#!/usr/bin/env bash

source ~/.config/kamera/bootstrap_app.sh
KAM_REPO_DIR=${KAM_REPO_DIR:-}
if [[ -z "${KAM_REPO_DIR}" ]]; then
    errcho "ERROR: Could not resolve KAM_REPO_DIR. Check ~/.config/kamera"
    exit 1
fi

source ${KAM_REPO_DIR}/src/cfg/cfg-aliases.sh

pids=("$@")
for host in $(cq '.hosts | keys | join("\n" )') ; do
    if [[ $(cq ".hosts.${host}.enabled") == 'true' ]]; then
        ssh_checked.sh $host "mount_nas ${ARGS[@]}" &
        pids[${i}]=$!
    else
        echo "${host} disabled."
    fi
done

for pid in ${pids[*]}; do
    wait $pid
done

declare -A RESULTS
declare -A EXITS
i=0
for host in $(cq '.hosts | keys | join("\n" )') ; do
    echo $i
    if [[ $(cq ".hosts.${host}.enabled") == 'true' ]]; then
        OUT=$(ssh_checked.sh $host "mount | grep kamera_nas")
        EXITS[${host}]=$?
        RESULTS[${host}]="$OUT"
    else
        echo "${host} disabled."
    fi
    i=$((i+1))
done

echo "results: ${RESULTS[@]}"
echo "exits: ${EXITS[@]}"


