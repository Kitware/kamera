#!/usr/bin/env bash
# Request host shutdown or reboot via kamerad HTTP API, then confirm the
# power action actually completed.
#
# A hung system (e.g. GPU fell off the bus) can keep answering ping while
# refusing SSH for a couple of minutes before it finally powers off, so
# confirmation is based on polling both ping and the SSH port and reporting
# the observed state until the host is fully down (and, for reboot, back up).
# Confirmation only applies to remote hosts -- when this machine (center)
# powers itself off there is nothing left to poll from.
#
# Usage:
#   kamera_power.sh shutdown left
#   kamera_power.sh reboot all
#   kamera_power.sh shutdown center0taiga

set -euo pipefail

ACTION=${1:?usage: kamera_power.sh <shutdown|reboot> <all|center|left|right|hostname>}
TARGET=${2:?usage: kamera_power.sh <shutdown|reboot> <all|center|left|right|hostname>}

# Seconds to wait for a host to go down / come back up before giving up.
SHUTDOWN_TIMEOUT=${KAMERA_SHUTDOWN_TIMEOUT:-600}
BOOT_TIMEOUT=${KAMERA_BOOT_TIMEOUT:-600}
POLL_INTERVAL=5

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

is_local_host() {
	local host=$1
	[[ "${host}" == "$(hostname)" || "${host}" == "$(hostname -s)" ]]
}

can_ping() {
	ping -c1 -W2 "$1" >/dev/null 2>&1
}

ssh_port_open() {
	timeout 3 bash -c ">/dev/tcp/$1/22" 2>/dev/null
}

kamera_power() {
	local host=$1
	local action=$2
	local response
	echo "Requesting ${action} on ${host} via kamerad..."
	# Fail fast if the host is unreachable (already off), but once connected
	# give kamerad plenty of time to respond -- it stops the supervisor
	# group (waiting) before triggering the power action.
	if ! response=$(curl -s --connect-timeout 5 --max-time 120 -X POST "http://${host}:8987/power/${action}"); then
		echo "ERROR: could not reach kamerad on ${host}" >&2
		return 1
	fi
	echo "kamerad response: ${response}"
	if [[ "${response}" != *'"ok": true'* && "${response}" != *'"ok":true'* ]]; then
		echo "ERROR: kamerad on ${host} did not accept the ${action} request" >&2
		return 1
	fi
	echo "OK: ${action} requested on ${host}"
}

wait_for_down() {
	local host=$1
	local start elapsed
	start=$(date +%s)
	echo "Waiting for ${host} to go down..."
	while true; do
		elapsed=$(( $(date +%s) - start ))
		if (( elapsed > SHUTDOWN_TIMEOUT )); then
			echo "ERROR: ${host} is still up after ${SHUTDOWN_TIMEOUT}s -- shutdown may have failed" >&2
			return 1
		fi
		if ! can_ping "${host}"; then
			# Require a second miss so one dropped packet doesn't count.
			sleep 2
			if ! can_ping "${host}"; then
				echo "${host} stopped responding to ping after ${elapsed}s"
				return 0
			fi
		fi
		if ssh_port_open "${host}"; then
			echo "  ${host}: still up, SSH responsive (${elapsed}s elapsed)"
		else
			echo "  ${host}: still pinging but SSH is unresponsive -- system is mid-shutdown or hung, this can take a couple of minutes (${elapsed}s elapsed)"
		fi
		sleep "${POLL_INTERVAL}"
	done
}

wait_for_up() {
	local host=$1
	local start elapsed
	start=$(date +%s)
	echo "Waiting for ${host} to come back up..."
	while true; do
		elapsed=$(( $(date +%s) - start ))
		if (( elapsed > BOOT_TIMEOUT )); then
			echo "ERROR: ${host} did not come back up within ${BOOT_TIMEOUT}s" >&2
			return 1
		fi
		if ssh_port_open "${host}"; then
			echo "${host} is accepting SSH again after ${elapsed}s"
			return 0
		fi
		if can_ping "${host}"; then
			echo "  ${host}: pinging, waiting for SSH (${elapsed}s elapsed)"
		else
			echo "  ${host}: no response yet (${elapsed}s elapsed)"
		fi
		sleep "${POLL_INTERVAL}"
	done
}

confirm_power() {
	local host=$1
	local action=$2
	if is_local_host "${host}"; then
		echo "${action} requested on this machine (${host}) -- it will go down momentarily"
		return 0
	fi
	case ${action} in
		shutdown)
			wait_for_down "${host}"
			echo "CONFIRMED: ${host} is fully shut down"
			;;
		reboot)
			wait_for_down "${host}"
			wait_for_up "${host}"
			echo "CONFIRMED: ${host} has fully rebooted"
			;;
	esac
}

pause_before_exit() {
	if [[ -t 0 && -t 1 ]]; then
		read -rp "Press enter to close..." || true
	fi
}

power_all() {
	local action=$1
	local failed=()
	local pos host

	# Request left/right first (best effort -- a host that is already off
	# should not prevent the others from powering down).
	for pos in left right; do
		host=$(kamera_host_for_position "${pos}")
		kamera_power "${host}" "${action}" || failed+=("${pos}: request failed")
	done
	for pos in left right; do
		host=$(kamera_host_for_position "${pos}")
		confirm_power "${host}" "${action}" || failed+=("${pos}: not confirmed")
	done

	if (( ${#failed[@]} )); then
		echo ""
		echo "WARNING: problems with the remote systems:" >&2
		printf '  %s\n' "${failed[@]}" >&2
		if [[ -t 0 && -t 1 ]]; then
			local reply
			read -rp "Continue with ${action} of center (this machine)? [y/N] " reply
			if [[ "${reply}" != [yY]* ]]; then
				echo "Leaving center up."
				return 1
			fi
		else
			echo "Leaving center up so the remote systems can be investigated." >&2
			return 1
		fi
	fi

	# Center goes last: it is the machine running this script.
	host=$(kamera_host_for_position center)
	kamera_power "${host}" "${action}"
	confirm_power "${host}" "${action}"
}

case "${TARGET}" in
	all)
		power_all "${ACTION}" || { pause_before_exit; exit 1; }
		;;
	center|left|right)
		host=$(kamera_host_for_position "${TARGET}")
		kamera_power "${host}" "${ACTION}" || { pause_before_exit; exit 1; }
		confirm_power "${host}" "${ACTION}" || { pause_before_exit; exit 1; }
		;;
	*)
		kamera_power "${TARGET}" "${ACTION}" || { pause_before_exit; exit 1; }
		confirm_power "${TARGET}" "${ACTION}" || { pause_before_exit; exit 1; }
		;;
esac

pause_before_exit
