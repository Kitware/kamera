#!/usr/bin/env bash
# Request host shutdown or reboot via kamerad HTTP API.
#
# Usage:
#   kamera_power.sh shutdown left
#   kamera_power.sh reboot all
#   kamera_power.sh shutdown center0taiga

set -euo pipefail

ACTION=${1:?usage: kamera_power.sh <shutdown|reboot> <all|center|left|right|hostname>}
TARGET=${2:?usage: kamera_power.sh <shutdown|reboot> <all|center|left|right|hostname>}

kamera_host_for_position() {
	local position=$1
	local system_name
	system_name=$(tr -d '[:space:]' < /home/user/kw/SYSTEM_NAME)
	case $position in
		center) echo "center0${system_name}" ;;
		left)   echo "left1${system_name}" ;;
		right)  echo "right2${system_name}" ;;
		*)
			echo "Unknown position: ${position} (expected center, left, or right)" >&2
			return 1
			;;
	esac
}

kamera_power() {
	local host=$1
	local action=$2
	echo "Requesting ${action} on ${host} via kamerad..."
	curl -sf -X POST "http://${host}:8987/power/${action}" || {
		echo "Failed to request ${action} on ${host}" >&2
		return 1
	}
	echo "OK: ${action} requested on ${host}"
    sleep 3
}

case "${TARGET}" in
	all)
		for pos in left right center; do
			kamera_power "$(kamera_host_for_position "${pos}")" "${ACTION}"
		done
		;;
	center|left|right)
		kamera_power "$(kamera_host_for_position "${TARGET}")" "${ACTION}"
		;;
	*)
		kamera_power "${TARGET}" "${ACTION}"
		;;
esac
