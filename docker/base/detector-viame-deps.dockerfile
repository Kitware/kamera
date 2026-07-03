# Build off the public VIAME docker build (with ITK support)
# NOTE (ROS2 port): Jazzy requires an Ubuntu 24.04 (noble) base. The VIAME
# image referenced here must be one built on 24.04; the old focal-based
# gpu-algorithms-seal tag cannot host Jazzy debs.
FROM kitware/viame:gpu-algorithms-seal AS vb

WORKDIR /root
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO jazzy
ENV DEBIAN_FRONTEND noninteractive

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    curl \
    git \
    vim \
    gnupg2 \
    jq \
    redis-server \
    iputils-ping \
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
    && rm -rf /var/lib/apt/lists/*

# Build tools necessary for catkin and roskv

# Add yq to make config query work
RUN curl -sL https://github.com/mikefarah/yq/releases/download/2.4.0/yq_linux_amd64 \
    -o /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq

## === === === === === === === === === === === === === === ===
# Clone in C++ Deps for Redis
RUN mkdir /src
RUN cd /src && git clone https://github.com/fmtlib/fmt.git \
    &&  mkdir -p /src/fmt/build \
    &&  cd /src/fmt/ \
    &&  git checkout 9c418bc468baf434a848010bff74663e1f820e79 \
    &&  cd /src/fmt/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=TRUE .. \
    &&  make -j && make install

RUN cd /src && git clone --depth 1 https://github.com/nlohmann/json.git \
    &&  mkdir -p /src/json/build && cd /src/json/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -D JSON_BuildTests=Off .. \
    &&  make -j && make install

RUN cd /src && git clone --depth 1 https://github.com/redis/hiredis.git \
    &&  mkdir -p /src/hiredis/build \
    &&  cd /src/hiredis/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release .. \
    &&  make -j && make install

RUN cd /src && git clone --depth 1 https://github.com/sewenew/redis-plus-plus.git \
    &&  mkdir -p /src/redis-plus-plus/build \
    &&  cd /src/redis-plus-plus/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -D REDIS_PLUS_PLUS_BUILD_TEST=OFF .. \
    &&  make -j && make install
