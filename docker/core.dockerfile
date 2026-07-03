## This should not invoke ANY internet calls!

FROM kamera/base/core-deps:latest

ENV REPO_DIR=/root/kamera
WORKDIR $REPO_DIR

COPY . $REPO_DIR

RUN rm -rf /entry \
    && ln -sf $REPO_DIR/src/run_scripts/entry /entry \
    && printf "\nsource /entry/project.sh\n" >> /root/.bashrc \
    && ln -sf $REPO_DIR/src/run_scripts/aliases.sh /aliases.sh \
    && printf "\nsource /aliases.sh\n" >> /root/.bashrc

RUN ln -sf $REPO_DIR/scripts/activate_ros.bash $REPO_DIR/activate_ros.bash
RUN ln -sf $REPO_DIR/src/cfg /cfg
RUN mkdir -p /root/.config/kamera && \
    ln -sf $REPO_DIR/.dir /root/.config/kamera/repo_dir.bash

# Build the ported ROS2 packages (backend pulls in the whole nayak/taiga set);
# unported ROS1 packages (phase_one, wxpython_gui, ...) are skipped by
# --packages-up-to and would not build under Jazzy anyway
RUN ln -sv /usr/bin/python3 /usr/bin/python || true
RUN [ "/bin/bash", "-c", "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --base-paths src --packages-up-to backend sprokit_adapters --cmake-args -DCMAKE_BUILD_TYPE=Release || colcon build --base-paths src --packages-up-to backend --cmake-args -DCMAKE_BUILD_TYPE=Release"]

ENTRYPOINT ["/entry/project.sh"]
CMD ["bash"]
