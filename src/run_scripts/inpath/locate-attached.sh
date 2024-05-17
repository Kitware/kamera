#!/usr/bin/env bash

# Locate an attached device's ip based on some ping chicanery
# Uses packet size of 55 to make filtering for the specific response more specific

errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

IFACE=${1}
if [[ -z "$IFACE" ]] ; then
    echo "must specify an interface as first arg"
    exit 1
fi

IFACE_IP=$(ipj.sh | jq -r ".${IFACE}.ip")
IFACE_BR=$(echo $IFACE_IP | sed -r -e 's/\.([0-9]+)\/[0-9]+/.255/g')
RE_PING_IP="(?<=63 bytes from )(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)"
RE_IP_SUB='^(\d{1,3}\.\d{1,3}\.\d{1,3})'


broadcast_scan() {
    BROAD_IP="${1}"
    ping -I "${IFACE}" -s 55 -w1 -c1 -b "${BROAD_IP}" 2>/dev/null | grep -P -o -e "${RE_PING_IP}"
    if [[ $? -ne 0 ]]; then
        errcho "Failed to locate device attached to $IFACE/$IFACE_IP @ ${BROAD_IP}"
        return 1
    fi
}
# Locate link-local devices
RES=$(broadcast_scan "$IFACE_BR" )
if [[ $? -ne 0 ]]; then
    errcho "Trying blast broadcast"
    RES=$(broadcast_scan "255.255.255.255" )
    if [[ $? -ne 0 ]]; then
        errcho "Failed to get response from any device on $IFACE"
        exit 1
    fi
fi


if [[ -z "$RES" ]]; then
    echo "Could not parse target ip from $IFACE/$IFACE_IP"
    exit 1
fi

IFACE_SUB=$(echo $IFACE_IP | grep -Po -e "${RE_IP_SUB}")
TARGET_SUB=$(echo $RES | grep -Po -e "${RE_IP_SUB}")

if [[ "$TARGET_SUB" != "$IFACE_SUB" ]]; then
    echo "Interface sub $IFACE/$IFACE_SUB did not match target IP $RES"
    exit 1
fi

echo $RES


