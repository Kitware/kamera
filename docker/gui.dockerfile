ARG BRANCH=latest
FROM kamera/base/kamera-gui-deps:latest


# Create a non-root user and switch to it. Running X11 applications as root does
# not always work.
#RUN adduser --uid 1000 --disabled-password --gecos '' --shell /bin/bash user
RUN useradd -m --uid=1000 user \
    && useradd --uid=7777 share \
    && usermod -aG share user

# Source code goes into ~/noaa_kamera, live dir goes into ~/kamera_ws
ENV HOME_DIR=/home/user \
    REPO_DIR=/home/user/noaa_kamera \
    WS_DIR=/home/user/kamera_ws \
    GUI_CFG_DIR=/home/user/.config/kamera/gui

# trying to speed up chown operation
RUN mkdir -p /home/user && chown -R user:user /home/user

COPY --chown=user:user  src/kitware-ros-pkg/wxpython_gui/config $GUI_CFG_DIR/config
COPY --chown=user:user  repo_dir.bash      $REPO_DIR/repo_dir.bash
COPY --chown=user:user  src                $REPO_DIR/src
COPY --chown=user:user  activate_ros.bash  $WS_DIR/activate_ros.bash
RUN ln -sf              $REPO_DIR/src      $WS_DIR/src                  &&\
    rm -rf /entry                                                       &&\
    ln -sf $REPO_DIR/src/run_scripts/entry /entry                       &&\
    printf "\nsource /entry/project.sh\n" >> /home/user/.bashrc         &&\
    touch $WS_DIR/.catkin_workspace                                     &&\
    ln -sf $REPO_DIR/src/run_scripts/aliases.sh /aliases.sh             &&\
    printf "\nsource /aliases.sh\n" >> /root/.bashrc                    &&\
    ln -sf $REPO_DIR/src/cfg /cfg                                       &&\
    HOME=/home/user $REPO_DIR/src/run_scripts/setup/install_links.sh

WORKDIR $WS_DIR
RUN find /home/user -not -user user -execdir chown user {} \+

# use the exec form of run because we need bash syntax
USER user
RUN [ "/bin/bash", "-c", "source /entry/project.sh && catkin build wxpython_gui "]
RUN [ "/bin/bash", "-c", "source /entry/project.sh && catkin build ins_driver "]
RUN [ "/bin/bash", "-c", "pip install -e ."]
USER root
RUN find /home/user -not -user user -execdir chown user {} \+
USER user

# Fast chown example:
# find /folder_with_lots_of_files -not -user someuser -execdir chown someuser {} \+

CMD ["bash"]
