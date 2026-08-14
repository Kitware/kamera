# This image contains the base of the ROS/CUDA for the system, plus
# a bunch of utility packages
# CUDA 13.0 = newest major with native support on the fleet's r580 drivers
# (13.1+ would rely on minor-version compatibility on r580 hosts; every node
# needs driver >= 580 before this image deploys).
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04 AS base_cuda_ubuntu

WORKDIR /root
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO jazzy
ENV DEBIAN_FRONTEND noninteractive

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# setup ROS2 apt source (Jazzy runs on Ubuntu 24.04 / noble)
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2-latest.list

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-core \
    ros-jazzy-ros-base \
    ros-jazzy-perception \
    python3-colcon-common-extensions \
    python3-pip \
    ros-jazzy-rqt-image-view \
    && rm -rf /var/lib/apt/lists/*

# ROS BUILD FINISHED

## Necessary, followed by unessential but useful packages
RUN     apt-key adv --keyserver hkps://keyserver.ubuntu.com --refresh-keys
RUN     apt-get update -q && apt-get install --no-install-recommends -y \
            curl \
            git \
            iputils-ping \
            iproute2 \
            net-tools \
            dnsutils \
            jq \
            rsync \
            fd-find \
            redis-tools \
            sqlite3 \
            python3-pip \
            python3-rosdep \
            unzip \
    &&  apt-get update -q && apt-get install --no-install-recommends -y \
            autoconf \
            automake\
            build-essential \
            dirmngr \
            pkg-config \
            sudo \
            nano \
            vim \
            inetutils-traceroute \
            tmux \
            python3-dev \
    &&  rm -rf /var/lib/apt/lists/*


## ipython isn't strictly required (like most things in is kitchen sink image) but it's extremely useful for debugging
RUN     pip install --break-system-packages --no-cache-dir \
            ipython \
            ipdb \
            pyserial \
            typing \
            pathlib \
            bottle \
            requests \
            profilehooks \
            redis \
            osrf-pycommon \
            six
