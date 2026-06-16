#!/bin/bash
# Copy start_stop .desktop shortcuts to a desktop folder and mark them launchable.
#
# GNOME requires both metadata::trusted and the executable bit before a .desktop
# file on the Desktop can be double-clicked without the "Allow launching" prompt.
# Order matters: set trusted first, then chmod +x.
#
# Usage:
#   ./install_desktop_shortcuts.sh              # installs to ~/Desktop
#   ./install_desktop_shortcuts.sh ~/Desktop    # custom destination

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SRC="${DIR}/desktop_shortcuts"
DEST="${1:-${HOME}/Desktop}"

if [[ ! -d "${SRC}" ]]; then
	echo "Missing shortcut directory: ${SRC}" >&2
	exit 1
fi

mkdir -p "${DEST}"

trust_desktop_file() {
	local file=$1
	if command -v gio >/dev/null 2>&1; then
		if [[ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
			gio set "${file}" metadata::trusted true
		else
			dbus-launch gio set "${file}" metadata::trusted true
		fi
	else
		echo "Warning: gio not found; ${file} may still prompt for trust" >&2
	fi
	chmod +x "${file}"
}

for shortcut in "${SRC}"/*.desktop; do
	[[ -e "${shortcut}" ]] || continue
	dest_file="${DEST}/$(basename "${shortcut}")"
	cp "${shortcut}" "${dest_file}"
	trust_desktop_file "${dest_file}"
	echo "Installed ${dest_file}"
done
