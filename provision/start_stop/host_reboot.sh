#!/usr/bin/env bash
# One-shot host reboot invoked by supervisor (user=root).
set -euo pipefail
exec /usr/bin/systemctl reboot
