# This image contains the base of the ROS/CUDA for the system, plus
# a bunch of utility packages
FROM nvidia/cuda:12.6.2-devel-ubuntu20.04 as base_cuda_ubuntu

WORKDIR /root
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic
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
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    ros-noetic-ros-base=1.5.0-1* \
    ros-noetic-perception=1.5.0-1* \
    python3-catkin-tools \
    python3-pip \
    ros-noetic-rqt-image-view \
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
            redis-tools \
            sqlite3 \
            python3-pip \
            python3-rosdep \
            python3-rosinstall \
            python3-vcstools \
            python3-catkin-tools \
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
RUN     pip install --upgrade --no-cache-dir pip \
    &&  pip install --no-cache-dir \
            ipython \
            ipdb \
            python-benedict \
            pyserial \
            typing \
            pathlib \
            bottle \
            requests \
            profilehooks \
            redis \
            osrf-pycommon \
            six
