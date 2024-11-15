# Build off the public VIAME docker build (with ITK support)
FROM kitware/viame:gpu-algorithms-seal as vb

WORKDIR /root
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic
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
    ros-noetic-rqt-image-view \
    && rm -rf /var/lib/apt/lists/*

# Build tools necessary for catkin and roskv

# Add Tini to handle signals
RUN curl -sSL https://github.com/krallin/tini/releases/download/v0.18.0/tini -o /tini && \
    chmod +x /tini

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
