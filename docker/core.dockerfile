## This should not invoke ANY internet calls!

FROM kamera/base/core-deps:latest

ENV REPO_DIR=/root/kamera
WORKDIR $REPO_DIR

COPY . $REPO_DIR

RUN rm -rf /entry \
    && ln -sf $REPO_DIR/src/run_scripts/entry /entry \
    && printf "\nsource /entry/project.sh\n" >> /root/.bashrc \
    && touch $REPO_DIR/.catkin_workspace \
    && ln -sf $REPO_DIR/src/run_scripts/aliases.sh /aliases.sh \
    && printf "\nsource /aliases.sh\n" >> /root/.bashrc

RUN ln -sf $REPO_DIR/scripts/activate_ros.bash $REPO_DIR/activate_ros.bash
RUN ln -sf $REPO_DIR/src/cfg /cfg
RUN mkdir -p /root/.config/kamera && \
    ln -sf $REPO_DIR/.dir /root/.config/kamera/repo_dir.bash

# Need to build phase_one first to generate SRV, then build backend
RUN ln -sv /usr/bin/python3 /usr/bin/python || true
RUN [ "/bin/bash", "-c", "source ${REPO_DIR}/activate_ros.bash && catkin build phase_one"]
RUN [ "/bin/bash", "-c", "source /entry/project.sh && catkin build -s backend"]

ENTRYPOINT ["/entry/project.sh"]
CMD ["bash"]
