ARG ROS_DISTRO
FROM kamera/base/roskitchen:${ROS_DISTRO}

RUN apt-get update && apt-get install --no-install-recommends -y \
        ros-${ROS_DISTRO}-compressed-image-transport\
        ros-${ROS_DISTRO}-camera-info-manager\
        ros-${ROS_DISTRO}-image-view \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-nodelet\
        ros-${ROS_DISTRO}-nodelet-topic-tools \
        libgtkmm-2.4-1v5 \
        libglademm-2.4-1v5 \
        libgtkglextmm-x11-1.2-dev \
        libgtkglextmm-x11-1.2-0v5 \
        libglade2-dev \
        libglademm-2.4-dev \
    && rm -rf /var/lib/apt/lists/*

## this will fail on noetic, that's ok, we can build from source
RUN apt-get update && apt-get install --no-install-recommends -y \
        ros-${ROS_DISTRO}-prosilica-camera\
        ros-${ROS_DISTRO}-vision-opencv\
        ros-${ROS_DISTRO}-prosilica-gige-sdk\
    || echo "failed to install camera packages" \
    && rm -rf /var/lib/apt/lists/*

## build deps

RUN     apt-get update -q && apt-get install --no-install-recommends -y \
            autoconf \
            automake\
            build-essential \
            dirmngr \
            unzip \
            pkg-config \
            udev \
            libudev-dev \
            libusb-1.0 \
            libhidapi-libusb0 \
            libtool \
            rsync \
            libhiredis-dev \
            nlohmann-json3-dev \
            usbutils \
    && rm -rf /var/lib/apt/lists/*

## ===================  install hid and DAQ drivers  ===================

WORKDIR /src
RUN curl -fsSL https://github.com/signal11/hidapi/archive/master.zip -o hidapi.zip
RUN curl -fsSL https://github.com/wjasper/Linux_Drivers/archive/master.zip -o mcc_drivers.zip
RUN unzip -q hidapi.zip &&\
    unzip -q mcc_drivers.zip -d mcc
WORKDIR /src/hidapi-master
RUN    ./bootstrap &&\
        ./configure &&\
        make -j`nproc` &&\
        make install

## ===================  other deps  ===================

WORKDIR /src

RUN :\
    &&  git clone https://github.com/fmtlib/fmt.git \
    &&  cd /src/fmt/ \
    &&  mkdir -p /src/fmt/build \
    &&  git checkout 9c418bc468baf434a848010bff74663e1f820e79 \
    &&  cd /src/fmt/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=TRUE -j .. \
    &&  make -j && make install \
    &&:

# todo: this is in-between patch releases but is necessary to build correctly. May need to fork and pin
RUN :\
    &&  curl -sSL https://github.com/sewenew/redis-plus-plus/archive/refs/tags/1.3.7.tar.gz -o 1.3.7.tar.gz \
    &&  tar -xzvf 1.3.7.tar.gz \
    &&  mkdir -p /src/redis-plus-plus-1.3.7/build \
    &&  cd /src/redis-plus-plus-1.3.7/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -D REDIS_PLUS_PLUS_BUILD_TEST=OFF -j .. \
    &&  make -j && make install \
    &&:

## === === === === === === Project specifics === === === === === === === ===
ENV REPO_DIR=/root/noaa_kamera \
    WS_DIR=/root/kamera_ws \
    CONTAINER_TYPE=kamera_base

WORKDIR /root/kamera_ws

RUN ln -sf   $REPO_DIR/src               $WS_DIR/src                    &&\
    rm -rf /entry                                                       &&\
    ln -sf $REPO_DIR/src/run_scripts/entry /entry                       &&\
    printf "\nsource /entry/project.sh\n" >> /root/.bashrc              &&\
    touch /root/kamera_ws/.catkin_workspace                             &&\
    ln -sf $REPO_DIR/src/run_scripts/aliases.sh /aliases.sh             &&\
    printf "\nsource /aliases.sh\n" >> /root/.bashrc                    &&\
    ln -sf $REPO_DIR/src/cfg /cfg

RUN     pip install --no-cache-dir \
            pyserial \
            osrf-pycommon \
            python-consul \
            python-benedict \
            six \
            redis \
            typing \
            shapely \
            pygeodesy \
            pyshp \
            scipy

WORKDIR /root/kamera_ws

COPY        src             $REPO_DIR/src
RUN [ "/bin/bash", "-c", "source /entry/project.sh && catkin build backend"]
RUN ln -sv /usr/bin/$PYTHON /usr/bin/python || true
COPY        scripts/activate_ros.bash           $WS_DIR/activate_ros.bash

ENTRYPOINT ["/entry/project.sh"]
CMD ["bash"]
