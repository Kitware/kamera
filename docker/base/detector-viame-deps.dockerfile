# Base image from which viame-kamera is built
# This may engage in some network access during the viame build
ARG VIAME_BRANCH=latest
FROM kamera/base/detector-base:${VIAME_BRANCH} as viame_src

RUN echo 'edit here to invalidate cache [0]' \
    && git clone https://github.com/VIAME/VIAME.git ${SRC_BASE}/viame \
    && cd ${SRC_BASE}/viame \
    && git checkout kamera/master \
    && git submodule update --init --recursive

FROM viame_src as viame_cache

# Throw checkpoint on this line so we can change hash without invalidating previous cache
ENV VIAME_CHECKPOINT 73df62910069f1de02e1abce8d78c8eb7f3fc40f
# Get VIAME - don't change this hash - this is for caching!
# kamera/master
#RUN echo 'edit here to invalidate cache [0]' \
#    && cd ${SRC_BASE}/viame \
#    && git checkout ${VIAME_CHECKPOINT} \
#    && git submodule update --init --recursive

# === === === === === ===  viame cache build === === === === ===
FROM viame_cache as viame_kamera

COPY src/run_scripts/setup/build_viame.sh       $REPO_DIR/src/run_scripts/setup/build_viame.sh

RUN VIAME_BRANCH=$VIAME_CHECKPOINT $REPO_DIR/src/run_scripts/setup/build_viame.sh

# now build latest kamera/master
#RUN $REPO_DIR/src/run_scripts/setup/build_viame.sh
