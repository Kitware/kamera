FROM kamera/base/viame:latest

COPY . /root/kamera

RUN ln -sf /root/kamera/src/run_scripts/entry /entry

WORKDIR /root/kamera
ENV REPO_DIR=/root/kamera
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

RUN ["/bin/bash", "-c", "source /entry/project.sh && \
      source src/run_scripts/setup/setup_viame_build.sh && \
      catkin build sprokit_adapters"]
