ARG CUDA
FROM kamera/base/core-ros-cuda${CUDA}:latest

RUN apt-get update && apt-get install --no-install-recommends -y \
        ros-noetic-compressed-image-transport \
        ros-noetic-camera-info-manager \
        ros-noetic-image-view \
        ros-noetic-cv-bridge \
        ros-noetic-nodelet \
        ros-noetic-nodelet-topic-tools \
        ros-noetic-vision-opencv \
        ros-noetic-diagnostic-updater \
        ros-noetic-self-test \
        libgtkmm-2.4-1v5 \
        libglademm-2.4-1v5 \
        libgtkglextmm-x11-1.2-dev \
        libgtkglextmm-x11-1.2-0v5 \
        libglade2-dev \
        libglademm-2.4-dev \
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

RUN     pip install --no-cache-dir \
            pyserial \
            osrf-pycommon \
            shapely \
            pygeodesy \
            pyshp \
            scipy

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

## === === === === === === Project specifics === === === === === === === ===
ENV WS_DIR=/root/noaa_kamera
WORKDIR /root/noaa_kamera

# Copy products into container
COPY        .             $WS_DIR

RUN rm -rf /entry                                                       &&\
    ln -sf $WS_DIR/src/run_scripts/entry /entry                        &&\
    printf "\nsource /entry/project.sh\n" >> /root/.bashrc              &&\
    touch $WS_DIR/.catkin_workspace                             &&\
    ln -sf $WS_DIR/src/run_scripts/aliases.sh /aliases.sh             &&\
    printf "\nsource /aliases.sh\n" >> /root/.bashrc

# Making useful links and copies
RUN ln -sf $WS_DIR/scripts/activate_ros.bash $WS_DIR/activate_ros.bash
RUN ln -sf $WS_DIR/src/cfg /cfg
RUN mkdir -p /root/.config/kamera && \
    ln -sf $WS_DIR/.dir /root/.config/kamera/repo_dir.bash

RUN [ "/bin/bash", "-c", "source /entry/project.sh && catkin build backend"]
RUN ln -sv /usr/bin/python3 /usr/bin/python || true

ENTRYPOINT ["/entry/project.sh"]
CMD ["bash"]
