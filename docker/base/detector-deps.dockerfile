# Base image from which the viame-base is built
# Most of the network access should occur here
# This should only rarely need to be rebuilt
FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as base

RUN apt-key del A4B469963BF863CC
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

ENV TZ="America/New_York"
ENV DEBIAN_FRONTEND noninteractive

# System Deps for VIAME
RUN apt-get update -q && apt-get install -y \
        zip \
        wget \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        git

# Need CMake >= 3.11.4 for VIAME,
# Thus, install latest stable CMake 3.14.1

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
    && apt-get update -y \
    && apt-get install -y \
        cmake \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libgl1-mesa-dev \
        libexpat1-dev \
        libgtk2.0-dev \
        libxt-dev \
        libxml2-dev \
        libssl-dev \
        liblapack-dev \
        openssl \
        python-dev \
        curl \
        python3-dev \
        dirmngr \
        gnupg2 \
        lsb-release \
        zlib1g-dev \
        cmake-curses-gui \
        python-pip \
        python-setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip==20.3.4\
     && pip install \
        numpy==1.15 \
        scipy==1.2 \
        six \
        pillow \
        enum34 \
        protobuf \
        pyyaml

# setup environment
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=melodic \
    REPO_DIR=/root/noaa_kamera \
    WS_DIR=/root/kamera_ws

# setup keys, then sources.list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

RUN apt-get update && apt-get install -y \
        ros-melodic-ros-base \
        ros-melodic-compressed-image-transport\
        ros-melodic-camera-info-manager\
        ros-melodic-camera-info-manager-py\
        ros-melodic-cv-bridge \
        ros-melodic-eigen-conversions \
        python-catkin-tools \
        ninja-build \
        curl \
        iputils-ping \
        dnsutils \
        redis-server \
        nano \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Add Tini to handle signals
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

# Add yq to make config query work
RUN curl -sL https://github.com/mikefarah/yq/releases/download/2.4.0/yq_linux_amd64 \
    -o /usr/local/bin/yq &&\
    chmod +x /usr/local/bin/yq

RUN curl -sL https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh \
    -o /root/Anaconda3-5.2.0-Linux-x86_64.sh  \
    && chmod +x /root/Anaconda3-5.2.0-Linux-x86_64.sh  \
    && /root/Anaconda3-5.2.0-Linux-x86_64.sh  -b -p /root/anaconda3 \
    && rm /root/Anaconda3-5.2.0-Linux-x86_64.sh

SHELL ["/bin/bash", "-c"]
# Get a separate version of opencv for venv
RUN source /root/anaconda3/bin/activate \
    && pip install --upgrade pip \
    && pip install opencv-python==4.3.0.38

# === === === === === ===  viame source code === === === === ===

# This part is really fruity. ITK takes FOREVER to build, and rarely changes
# However, the ITK configuration is pegged to VIAME. Therefore, to allow us
# to cache aggressively while still having the ability to update the Viame
# source code, we will create a stage which builds Viame at a fixed point,
# update the source code, and copy in the build tree. I'm sorry future everyone
# maintaining this.
FROM base as viame_src

WORKDIR /src

# Treat these like static variables. It just makes copy-pasting scripts easier.
ENV BUILD_DIR=/build \
    SRC_BASE=/src \
    REPO_DIR=/src/noaa_kamera \
    PROJ_NAME=viame

RUN mkdir -p ${SRC_BASE}/${PROJ_NAME} ${BUILD_DIR}/${PROJ_NAME} /release

## === === === === === === === === === === === === === === ===
# Clone in C++ Deps for Redis
WORKDIR /src
RUN     git clone https://github.com/fmtlib/fmt.git \
    &&  mkdir -p /src/fmt/build \
    &&  cd /src/fmt/\
    &&  git checkout 9c418bc468baf434a848010bff74663e1f820e79 \
    &&  cd /src/fmt/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=TRUE .. \
    &&  make -j && make install \
    &&:
RUN     git clone --depth 1 https://github.com/nlohmann/json.git \
    &&  mkdir -p /src/json/build && cd /src/json/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -D JSON_BuildTests=Off .. \
    &&  make -j && make install \
    &&:
RUN git clone --depth 1 https://github.com/redis/hiredis.git \
    &&  mkdir -p /src/hiredis/build \
    &&  cd /src/hiredis/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release .. \
    &&  make -j && make install \
    &&:
RUN :\
    &&  git clone --depth 1 https://github.com/sewenew/redis-plus-plus.git \
    &&  mkdir -p /src/redis-plus-plus/build \
    &&  cd /src/redis-plus-plus/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -D REDIS_PLUS_PLUS_BUILD_TEST=OFF .. \
    &&  make -j && make install \
    &&:
## === === === === === === === === === === === === === === ===
