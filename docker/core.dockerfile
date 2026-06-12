## This should not invoke ANY internet calls!

FROM kamera/base/core-deps:latest

ENV WS_DIR=/root/kamera
WORKDIR $WS_DIR

COPY . $WS_DIR

RUN rm -rf /entry \
    && ln -sf $WS_DIR/src/run_scripts/entry /entry \
    && printf "\nsource /entry/project.sh\n" >> /root/.bashrc \
    && touch $WS_DIR/.catkin_workspace \
    && ln -sf $WS_DIR/src/run_scripts/aliases.sh /aliases.sh \
    && printf "\nsource /aliases.sh\n" >> /root/.bashrc

RUN ln -sf $WS_DIR/scripts/activate_ros.bash $WS_DIR/activate_ros.bash
RUN ln -sf $WS_DIR/src/cfg /cfg
RUN mkdir -p /root/.config/kamera && \
    ln -sf $WS_DIR/.dir /root/.config/kamera/repo_dir.bash

# Need to build phase_one first to generate SRV, then build backend
RUN ln -sv /usr/bin/python3 /usr/bin/python || true
RUN [ "/bin/bash", "-c", "source ${WS_DIR}/activate_ros.bash && catkin build phase_one"]
RUN [ "/bin/bash", "-c", "source /entry/project.sh && catkin build -s backend"]

ENTRYPOINT ["/entry/project.sh"]
CMD ["bash"]
