#!/usr/bin/env bash
## Input variables
## image:tag or uri/image:tag
URITAG=${URITAG}
## Image (without :tag)
IMGURI=${IMGURI}
## Tag provided separately
TAG=${TAG}

## Docker pseudo-file cache path
DOCKER_CACHE=${DOCKER_CACHE:-.cache/image}

## constants
PAT_BUILT_HASH='(?<=writing image sha256\:|Successfully built )([a-f0-9]{12})'
PAT_BUILT_IMG='(?<=Successfully tagged )(.+)'

errcho() {
  (echo >&2 -e "\e[31m$1\e[0m")
}

if ! echo hi | grep -P hi &>/dev/null; then
  errcho "Error: You need gnu grep (with -P flag) in your PATH"; exit 1
fi

## construct URITAG from IMGURI and TAG if URITAG not provided
_URITAG=${TAG:+${IMGURI}:${TAG}}
_URITAG=${_URITAG:-${IMGURI}}
URITAG=${URITAG:-${_URITAG}}

## split the finalized URITAG into parts
IMGURI=${URITAG%%:*}
if [[ ${URITAG} == *":"* ]]; then
  TAG=${URITAG##*:}
else
  TAG=${TAG:-__unset__}
fi

DFILE_DIR="${DOCKER_CACHE}/${IMGURI}"
DFILE_FN="${DFILE_DIR}/${TAG}"
mkdir -p "${DFILE_DIR}"

( echo >&2 -e '\e[36;1m$\e[35m' docker build ${URITAG:+-t ${URITAG}} "$@" "\e[0m")
docker build ${URITAG:+-t ${URITAG}} "$@" 2>&1 | tee >( grep -Po "${PAT_BUILT_HASH}" > "${DFILE_FN}" )
echo $DFILE_FN
