#!/usr/bin/env bash

## Aliases used with configurating

## make this config dir self-hosting directory-wise
CFG_DIR="${KAM_REPO_DIR}/src/cfg"
export CFG_ALIAS_SET=true

# ConfigQuery
cq () {
    CFG_FILE=${CFG_DIR}/${SYSTEM_NAME}/config.yaml ${CFG_DIR}/get "$@"
}
