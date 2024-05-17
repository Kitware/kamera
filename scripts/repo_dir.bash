#!/usr/bin/env bash

## Use this file to initialize a path variable to this location on the current filesystem
## Usage: load into a variable in a script, like this:
##     THIS_DIR=$(/abs/or/link/path/to/repo_dir.bash)
## For best results, invoke directly, e.g. `/abs/path/repo_dir.bash`,
## not `source repo_dir.bash` or `bash repo_dir.bash`

HEREDIR=$(cd "$(dirname $(realpath ${0}))" && pwd)
echo ${HEREDIR}
