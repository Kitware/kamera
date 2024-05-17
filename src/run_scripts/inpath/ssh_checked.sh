#!/usr/bin/env bash

# Check connection to TARGET_HOST. If successful, execute remote command on it
errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

blueprintf() {
    (printf "\e[35m$@\e[0m")
}

resetcolor() {
    (printf "\e[0m")
}

if [[ -z "$@" ]] ; then
    errcho "No arguments provided"
    exit 1
fi

TARGET_HOST="$1"
ARGS=( "${@:2}" )

blueprintf "$TARGET_HOST\n"

PINGRES=$(ping -c1 -W1 ${TARGET_HOST})
STATUS=$?
echo "${PINGRES}" | grep -P -e '^.+bytes from'
blueprintf "$ ${ARGS[@]}"
echo ""
resetcolor
echo '---'
if [[ $STATUS -eq 0 ]] ; then
    res=$(ssh ${TARGET_HOST} "${ARGS[@]}")
    STATUS=$?
    printf "\e[32m"
    echo "${res}" | sed -n ":a; $ ! {N;ba}; s/\n/\n${TARGET_HOST}| /g;p"
    printf "\e[0m\e[0m"
    if [[ $STATUS -ne 0 ]]; then
        exit $STATUS
    fi
else
    errcho "Unable to ping ${TARGET_HOST}"
    exit 63
fi

