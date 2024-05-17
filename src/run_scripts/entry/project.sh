#!/bin/bash

# General project entrypoint. Note the `exec` ! All subprocesses must be forked to
# properly handle signals.
source /entry/project_env.sh
exec "$@"
