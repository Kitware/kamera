#!/usr/bin/env bash
# One-shot host shutdown invoked by supervisor (user=root).
#
# The power action is deferred a few seconds via a transient systemd timer so
# this script exits 0 immediately and kamerad can send its HTTP reply before
# the host starts tearing down services (otherwise the caller sees the
# connection drop and reports a spurious "could not reach kamerad").
set -euo pipefail
DELAY=${KAMERA_POWER_DELAY:-5}
if command -v systemd-run >/dev/null 2>&1; then
	exec systemd-run --quiet --on-active="${DELAY}" \
		--timer-property=AccuracySec=1s /sbin/shutdown -h now
fi
# Fallback without systemd: detach so the one-shot still returns promptly.
nohup bash -c "sleep ${DELAY}; exec /sbin/shutdown -h now" >/dev/null 2>&1 &
