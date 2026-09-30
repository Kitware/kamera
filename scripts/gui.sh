#!/usr/bin/env bash

## `kamera gui`: open the control panel on a running system. Stays in the
## foreground until the GUI closes.
export COMPOSE_IGNORE_ORPHANS=True # make compose quieter
KAM_REPO_DIR=${KAM_REPO_DIR:-$(/home/user/.config/kamera/repo_dir.bash)}
source "${KAM_REPO_DIR}/runtime/env.sh"

if [[ $# -gt 0 ]]; then
    echo "usage: kamera gui" >&2
    exit 2
fi

notify-send -t 5000 "KAMERA" "Starting KAMERA Control Panel, please wait" || true
cd "${KAM_REPO_DIR}"
xhost +local:root
exec docker compose -f "${KAM_REPO_DIR}/compose/gui.yml" up
