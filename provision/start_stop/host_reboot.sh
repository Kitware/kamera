#!/usr/bin/env bash
# One-shot host reboot invoked by supervisor (user=root).
#
# Deferred a few seconds via a transient systemd timer so kamerad can send
# its HTTP reply before the host starts going down (see host_shutdown.sh).
set -euo pipefail
DELAY=${KAMERA_POWER_DELAY:-5}
if command -v systemd-run >/dev/null 2>&1; then
	exec systemd-run --quiet --on-active="${DELAY}" \
		--timer-property=AccuracySec=1s /usr/bin/systemctl reboot
fi
nohup bash -c "sleep ${DELAY}; exec /usr/bin/systemctl reboot" >/dev/null 2>&1 &
