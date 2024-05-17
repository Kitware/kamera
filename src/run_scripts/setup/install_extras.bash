#!/bin/bash
# Sets up the Nuvo computer from a fresh Ubuntu X.04 install

sudo apt update
sudo apt install -y \
    jq \
    lm-sensors \
    tree \
    cowsay \
    lldpd

# get yq
sudo curl -sL https://github.com/mikefarah/yq/releases/download/2.4.0/yq_linux_amd64 \
    -o /usr/local/bin/yq &&\
    sudo chmod +x /usr/local/bin/yq

