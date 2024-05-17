#!/usr/bin/env bash

errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}


TARGET="${1}"

if [[ -z "$TARGET" ]]; then
    errcho "Must provide a host/IP target. Usage: fastping TARGET [TIMEOUT]"
    exit 1
fi

TIMEOUT="${2:-0.1}"

exec sudo python3 -c \
"\
import ping3, sys, socket;
target = str(\"${TARGET}\")
timeout = float(${TIMEOUT})
res = ping3.ping(target, timeout=timeout, unit='ms');
if res is None:
  ts = 'failed or timed out'
  sys.stderr.write('ping {}: failed or timed out ({}s)\n'.format(target, timeout))
  sys.stderr.flush()
  sys.exit(1)
ts = '{:.1f}ms'.format(res)
print('{: <20} ({: <16}): {: >8}'.format(target, socket.gethostbyname(target), ts));
sys.exit(not res)"
