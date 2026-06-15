ARG BRANCH=latest
ARG GUI_DEPS_IMAGE=kamera/base/core-gui-deps:latest
FROM ${GUI_DEPS_IMAGE}


# Create a non-root user and switch to it. Running X11 applications as root does
# not always work.
#RUN adduser --uid 1000 --disabled-password --gecos '' --shell /bin/bash user
RUN useradd -m --uid=1000 user \
    && useradd --uid=7777 share \
    && usermod -aG share user

# Source code goes into ~/noaa_kamera, live dir goes into ~/kamera_ws
ENV HOME_DIR=/home/user \
    REPO_DIR=/home/user/kamera \
    GUI_CFG_DIR=/home/user/.config/kamera/gui

# trying to speed up chown operation
RUN mkdir -p /home/user && chown -R user:user /home/user

COPY --chown=user:user  .                        $REPO_DIR
RUN ln -sf $REPO_DIR/src/run_scripts/entry /entry                       &&\
    printf "\nsource /entry/project.sh\n" >> /home/user/.bashrc         &&\
    touch $REPO_DIR/.catkin_workspace                                     &&\
    ln -sf $REPO_DIR/src/cfg /cfg

WORKDIR $REPO_DIR
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
