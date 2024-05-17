# This adds the KAMERA code and builds the SprokitAdapter

ARG VIAME_BRANCH=latest
FROM kamera/base/viame-base:${VIAME_BRANCH}

ENV WS_DIR=/root/kamera_ws

WORKDIR /root/kamera_ws

COPY repo_dir.bash                              $REPO_DIR/repo_dir.bash
COPY src/kitware-ros-pkg/rqt_sprokit_adapter    $REPO_DIR/src/kitware-ros-pkg/rqt_sprokit_adapter
COPY src/kitware-ros-pkg/sprokit_adapters       $REPO_DIR/src/kitware-ros-pkg/sprokit_adapters
COPY src/process/nexus                          $REPO_DIR/src/process/nexus
COPY src/core/roskv                             $REPO_DIR/src/core/roskv
COPY src/custom_msgs                            $REPO_DIR/src/custom_msgs
COPY src/run_scripts                            $REPO_DIR/src/run_scripts
COPY src/cfg                                    $REPO_DIR/src/cfg
COPY scripts/activate_ros.bash                  $WS_DIR/activate_ros.bash

RUN ln -sf $REPO_DIR/src                        $WS_DIR/src    &&\
    rm -rf /entry                                              &&\
    ln -sf $REPO_DIR/src/run_scripts/entry      /entry         &&\
    printf "\nsource /entry/project.sh\n" >> /root/.bashrc     &&\
    touch $WS_DIR/.catkin_workspace                            &&\
    ln -sf $REPO_DIR/src/run_scripts/aliases.sh /aliases.sh    &&\
    printf "\nsource /aliases.sh\n" >> /root/.bashrc           &&\
    ln -sf $REPO_DIR/src/cfg                    /cfg           &&\
    bash $REPO_DIR/src/run_scripts/setup/install_links.sh

RUN $REPO_DIR/src/run_scripts/setup/build_viame.sh

# use the exec form of run because we need bash syntax
RUN [ "/bin/bash", "-c","source /entry/project.sh && catkin build roskv"]
RUN [ "/bin/bash", "-c","source /entry/project.sh && catkin build"]
RUN [ "/bin/bash", "-c", "pip install -e ."]
#ENTRYPOINT ["/entry/project.sh"]
CMD ["bash"]
