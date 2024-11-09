FROM kamera/base/viame:latest

COPY . /root/noaa_kamera

RUN ln -sf /root/noaa_kamera/src/run_scripts/entry /entry

WORKDIR /root/noaa_kamera
ENV REPO_DIR=/root/noaa_kamera \
    WS_DIR=/root/noaa_kamera

RUN ["/bin/bash", "-c", "source /entry/project.sh && \
      source src/run_scripts/setup/setup_viame_build.sh && \
      catkin build sprokit_adapters"]
