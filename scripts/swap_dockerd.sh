#!/usr/bin/bash

KAM_REPO_DIR=$(/home/user/.config/kamera/repo_dir.bash)
echo $KAM_REPO_DIR
SYSTEM_NAME=$(cat /home/user/kw/SYSTEM_NAME)
source "${KAM_REPO_DIR}/tmux/${SYSTEM_NAME}/env.sh"

MASTER_HOST=$(cq '.master_host')

sudo service docker stop
sudo dockerd --insecure-registry ${MASTER_HOST}:5000
