## This should not invoke ANY internet calls!

ARG ROS_DISTRO
FROM kamera/base/kamera-deps-${ROS_DISTRO}:latest

WORKDIR $REPO_DIR
COPY        .             $REPO_DIR

# use the exec form of run because we need bash syntax
RUN [ "/bin/bash", "-c", "source /entry/project.sh && catkin build backend"]
RUN [ "/bin/bash", "-c", "pip install ."]


ENTRYPOINT ["/entry/project.sh"]
CMD ["bash"]
