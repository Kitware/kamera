#!/usr/bin/env bash
# One-shot host shutdown invoked by supervisor (user=root).
set -euo pipefail
exec /sbin/shutdown -h now
