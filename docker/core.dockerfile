## This should not invoke ANY internet calls!

FROM kamera/base/core-deps:latest

COPY . /root/kamera
WORKDIR /root/kamera

# use the exec form of run because we need bash syntax
RUN [ "/bin/bash", "-c", "source /entry/project.sh && catkin build -s backend"]
# python package is currently only supported for python3.10+, not on ubuntu20.04
# RUN [ "/bin/bash", "-c", "pip install ."]



ENTRYPOINT ["/entry/project.sh"]
CMD ["bash"]
