FROM ros:noetic-robot as noetic-robot
ENV PYTHON=python3 \
    ROS_PKG_VERS=1.5.0-1

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
            profilehooks \
            redis \
            osrf-pycommon \
            six
