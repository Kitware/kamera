#!/bin/bash
# Sets up the Nuvo computer from a fresh Ubuntu X.04 install

sudo apt update
sudo apt install -y \
    "git" \
    "zsh" \
    "tmux" \
    "curl" \
    "wget" \
    "nmap" \
    "htop" \
    "arp-scan" \
    traceroute \
    openssh-server \
    apt-transport-https \
    ca-certificates \
    gnupg-agent \
    build-essential \
    pkg-config \
    software-properties-common

# docker repo keys
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
	"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
	$(lsb_release -cs) stable"

sudo apt update
sudo apt install -y \
	docker-ce


# Create the docker group
sudo groupadd docker
# add the current user to the docker group
sudo usermod -aG docker $USER

# Docker compose
sudo curl -L https://github.com/docker/compose/releases/download/1.24.0/docker compose-`uname -s`-`uname -m` -o /usr/local/bin/docker compose
sudo chmod +x /usr/local/bin/docker compose

