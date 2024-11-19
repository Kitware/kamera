#!/usr/bin/bash

IMAGES=(\
        "kitware/kamera:viame"
        "kitware/kamera:core"
        "kitware/kamera:postproc"
        "kitware/kamera:kamerad"
       )

KAM_REPO_DIR=$(/home/user/.config/kamera/repo_dir.bash)
echo $KAM_REPO_DIR
SYSTEM_NAME=$(cat /home/user/kw/SYSTEM_NAME)
source "${KAM_REPO_DIR}/tmux/${SYSTEM_NAME}/env.sh"

MASTER_HOST=$(cq '.master_host')

if [[ -z "$@" ]] ; then
    ARGS=(start)
else
    ARGS=("$@")
fi

OP=""
for arg in "${ARGS[@]}" ; do
    echo "arg: ${arg}"
    if [[ "${arg}" == "push"      ]] ; then OP=push ; fi
    if [[ "${arg}" == "pull"      ]] ; then OP=pull ; fi
    if [[ "${arg}" == "local"    ]] ; then SERVER="local" ; fi
    if [[ "${arg}" == "remote" ]] ; then SERVER="remote" ; fi
done

if [[ -z "${OP}" ]]; then
    errcho "Missing docker operation (push/pull)"
    exit 1
fi

if [[ -z "${SERVER}" ]]; then
  errcho "Missing server to push/pull from (local/remote)"
    exit 1
fi

# Synchronize local images to either the remote or local repositories

# if local, pull/push from the master host
if [[ $SERVER == "local" ]]; then
  ORIGIN="${MASTER_HOST}:5000"
  if [[ $OP == "push" ]]; then
    for img in "${IMAGES[@]}" ; do
      docker tag $img ${ORIGIN}/${img}
      docker push ${ORIGIN}/${img}
    done
  fi
  if [[ $OP == "pull" ]]; then
    for img in "${IMAGES[@]}" ; do
      docker pull ${ORIGIN}/${img}
      docker tag ${ORIGIN}/${img} ${img}
    done
  fi
fi

# if remote, direct to docker hub
if [[ $SERVER == "remote" ]]; then
  if [[ $OP == "push" ]]; then
    for img in "${IMAGES[@]}" ; do
      docker push ${img}
    done
  fi
  if [[ $OP == "pull" ]]; then
    for img in "${IMAGES[@]}" ; do
      docker pull ${img}
    done
  fi
fi
