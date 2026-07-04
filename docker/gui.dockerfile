ARG BRANCH=latest
ARG GUI_DEPS_IMAGE=kamera/base/gui-deps:latest
FROM ${GUI_DEPS_IMAGE}


# Create a non-root user and switch to it. Running X11 applications as root does
# not always work.
#RUN adduser --uid 1000 --disabled-password --gecos '' --shell /bin/bash user
RUN useradd -m --uid=1000 user \
    && useradd --uid=7777 share \
    && usermod -aG share user

# Source code goes into ~/kamera
ENV HOME_DIR=/home/user \
    REPO_DIR=/home/user/kamera \
    GUI_CFG_DIR=/home/user/.config/kamera/gui

# trying to speed up chown operation
RUN mkdir -p /home/user && chown -R user:user /home/user

WORKDIR $REPO_DIR
COPY --chown=user:user  .  $REPO_DIR

RUN rm -rf /entry \
    && ln -sf $REPO_DIR/src/run_scripts/entry /entry \
    && printf "\nsource /entry/project.sh\n" >> /home/user/.bashrc \
    && ln -sf $REPO_DIR/src/run_scripts/aliases.sh /aliases.sh \
    && printf "\nsource /aliases.sh\n" >> /home/user/.bashrc

RUN ln -sf $REPO_DIR/scripts/activate_ros.bash $REPO_DIR/activate_ros.bash
RUN ln -sf $REPO_DIR/src/cfg /cfg
RUN mkdir -p /home/user/.config/kamera && \
    ln -sf $REPO_DIR/.dir /home/user/.config/kamera/repo_dir.bash

RUN ln -sv /usr/bin/python3 /usr/bin/python || true
RUN find /home/user -not -user user -execdir chown user {} \+

# Install kamera for wxpython_gui imports. --no-deps: deps come from the base
# image (a full install trips on ROS's distutils PyYAML). --break-system-packages:
# Ubuntu 24.04 marks its python as externally managed. Jazzy's python 3.12 clears
# our 3.10 floor, so the Noetic-era --ignore-requires-python is gone.
RUN pip install --break-system-packages --no-cache-dir matplotlib \
    && pip install --break-system-packages --no-cache-dir --no-deps -e $REPO_DIR

# use the exec form of run because we need bash syntax
USER user
RUN [ "/bin/bash", "-c", "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --base-paths src --packages-up-to wxpython_gui ins_driver --cmake-args -DCMAKE_BUILD_TYPE=Release"]
USER root
RUN find /home/user -not -user user -execdir chown user {} \+
USER user

# Fast chown example:
# find /folder_with_lots_of_files -not -user someuser -execdir chown someuser {} \+

CMD ["bash"]
