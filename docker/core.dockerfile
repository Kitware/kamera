## This should not invoke ANY internet calls!

ARG CUDA
FROM kamera/base/core-deps-cuda${CUDA}:latest

WORKDIR $WS_DIR
COPY . $WS_DIR

# use the exec form of run because we need bash syntax
RUN [ "/bin/bash", "-c", "source /entry/project.sh && catkin build backend"]
# python package is currently only supported for python3.10+, not on ubuntu20.04
# RUN [ "/bin/bash", "-c", "pip install ."]



ENTRYPOINT ["/entry/project.sh"]
CMD ["bash"]
