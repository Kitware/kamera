FROM kamera/base/core-ros:latest

RUN apt-get update && apt-get install --no-install-recommends -y \
        ros-jazzy-compressed-image-transport \
        ros-jazzy-image-transport \
        ros-jazzy-camera-info-manager \
        ros-jazzy-camera-calibration-parsers \
        ros-jazzy-cv-bridge \
        ros-jazzy-vision-opencv \
        ros-jazzy-diagnostic-updater \
        ros-jazzy-rosidl-default-generators \
    && rm -rf /var/lib/apt/lists/*

## build deps

RUN apt-get update -q && apt-get install --no-install-recommends -y \
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
            libhiredis-dev \
            nlohmann-json3-dev \
            usbutils \
    && rm -rf /var/lib/apt/lists/*

# --ignore-installed: scipy/shapely pull numpy, which pip otherwise tries (and
# fails) to uninstall from the Debian-owned site-packages on Ubuntu 24.04.
# numpy stays <2: the distro python3-opencv/cv_bridge are built against the
# numpy 1 ABI and "import cv2" fails under numpy 2.
RUN     pip install --break-system-packages --no-cache-dir --ignore-installed \
            "numpy<2" \
            pyserial \
            osrf-pycommon \
            shapely \
            pygeodesy \
            pyshp \
            scipy

## ===================  install hid and DAQ drivers  ===================

WORKDIR /src
# signal11/hidapi is unmaintained and its autotools setup no longer bootstraps
# under Ubuntu 24.04's autoconf; the Debian package provides the same
# libhidapi-libusb + headers the MCC DAQ drivers link against.
RUN apt-get update && apt-get install -q -y --no-install-recommends \
        libhidapi-dev libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://github.com/wjasper/Linux_Drivers/archive/master.zip -o mcc_drivers.zip
RUN unzip -q mcc_drivers.zip -d mcc

## ===================  other deps  ===================

WORKDIR /src

RUN :\
    &&  git clone https://github.com/fmtlib/fmt.git \
    &&  cd /src/fmt/ \
    &&  mkdir -p /src/fmt/build \
    &&  git checkout 9c418bc468baf434a848010bff74663e1f820e79 \
    &&  cd /src/fmt/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=TRUE .. \
    &&  make -j && make install \
    &&:

# todo: this is in-between patch releases but is necessary to build correctly. May need to fork and pin
RUN :\
    &&  curl -sSL https://github.com/sewenew/redis-plus-plus/archive/refs/tags/1.3.7.tar.gz -o 1.3.7.tar.gz \
    &&  tar -xzvf 1.3.7.tar.gz \
    &&  sed -i '0,/#include/s//#include <cstdint>\n&/' /src/redis-plus-plus-1.3.7/src/sw/redis++/utils.h \
    &&  mkdir -p /src/redis-plus-plus-1.3.7/build \
    &&  cd /src/redis-plus-plus-1.3.7/build \
    &&  cmake -D CMAKE_BUILD_TYPE=Release -D REDIS_PLUS_PLUS_BUILD_TEST=OFF .. \
    &&  make -j && make install \
    &&:

# Add yq for configu query to work
RUN  curl -sL https://github.com/mikefarah/yq/releases/download/3.4.1/yq_linux_amd64 \
     -o /usr/local/bin/yq && \
     chmod +x /usr/local/bin/yq

## ===================  install ebus sdk  ===================
COPY ./artifacts/ebus.deb /ebus.deb
COPY ./artifacts/GigE-V-Framework_x86_2.02.0.0132.tar.gz /gigev.tar.gz
RUN dpkg -i /ebus.deb && rm /ebus.deb \
    &&  tar xf /gigev.tar.gz && mkdir -p /src && rm /gigev.tar.gz
RUN cd /src/DALSA &&\
    sed -re 's/read -p.*$//g' GigeV/bin/install.gigev |\
    sed -re 's/^\s*\$OUTPUT_LICENSE$//g' |\
    sed -re 's/INSTALL_PROMPT=""/INSTALL_PROMPT="Accept."/g' \
    > GigeV/bin/temp && mv GigeV/bin/temp GigeV/bin/install.gigev &&\
    chmod +x GigeV/bin/install.gigev && ./corinstall install

