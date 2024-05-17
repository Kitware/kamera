## Base image for ROS with all the helpful fixin's. Everything + the kitchen sink
## This should be comparatively stable and used as a base image for most non-Nvidia containers
## This takes ROS_DISTRO as a primary build parameter, {kinetic, melodic, noetic}
## PYTHON is determined based on the ros distro, that's what this switch is based off of

ARG ROS_DISTRO
FROM ros:kinetic-robot as kinetic-robot
ENV PYTHON=python \
    ROS_PKG_VERS=1.3.2-0

FROM ros:melodic-robot as melodic-robot
ENV PYTHON=python \
    ROS_PKG_VERS=1.4.1-0

FROM ros:noetic-robot as noetic-robot
ENV PYTHON=python3 \
    ROS_PKG_VERS=1.5.0-1

## === === === === === === === ros desktop === === === === === === === === === ===
## this is the switch which pulls from one of the above stages, depending on ROS_DISTRO.
FROM ${ROS_DISTRO}-robot as ros-desktop
# bug: https://askubuntu.com/questions/650032/gpg-errorthe-following-signatures-were-invalid-keyexpired
RUN     apt-key adv --keyserver hkps://keyserver.ubuntu.com --refresh-keys
RUN     apt-get update \
    &&  apt-get install -y curl git jq sudo \
    &&  apt-get install -y --no-install-recommends \
            ros-${ROS_DISTRO}-desktop=${ROS_PKG_VERS}* \
    &&  rm -rf /var/lib/apt/lists/*

## === === === === === === === ros desktop plus everything else === === === === === === === === === ===
FROM ros-desktop as ros-desktop-plus

# Add yq to make config query work
# Add Tini to handle signals. This is mostly outdated but some entrypoints still use it
COPY --from=kamera/base/tini:v0190 /bin/tini /bin/tini
## Package of useful static binaries
COPY --from=kamera/base/static-bin-utils:latest /usr/local/bin/* /usr/local/bin/

RUN     :\
    &&  chmod +x /usr/local/bin/* \
    &&  chmod +x /bin/tini \
    &&:

## Necessary, followed by unessential but useful packages
RUN     printf "\n======\nenv: ROS_DISTRO: ${ROS_DISTRO} PYTHON: ${PYTHON}\n======\n" \
    &&  apt-get update -q && apt-get install --no-install-recommends -y \
            curl \
            git \
            iputils-ping \
            iproute2 \
            net-tools \
            dnsutils \
            jq \
            redis-tools \
            sqlite3 \
            ${PYTHON}-pip \
            ${PYTHON}-rosdep \
            ${PYTHON}-rosinstall \
            ${PYTHON}-vcstools \
            ${PYTHON}-catkin-tools \
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
RUN     pip install --upgrade --no-cache-dir "pip~=20.3" \
    &&  pip install --no-cache-dir \
            ipython \
            ipdb \
            python-benedict \
            pyserial \
            typing \
            pathlib \
            profilehooks \
            redis \
            osrf-pycommon \
            six
