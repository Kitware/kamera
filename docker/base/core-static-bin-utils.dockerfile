# using busybox or similar keeps the rust binaries from working, due to glibc
# This is only for use as data-only container. To have them as runnable entrypoints, you
# need a base with glibc
ARG FINAL_BASE=busybox
# Ubuntu base for building things with gcc, cmake and the like

FROM ubuntu:bionic as base

RUN     apt-get update -qq \
    &&  apt-get install -qq \
            apt-transport-https \
            autoconf \
            bison \
            build-essential \
            ca-certificates \
            curl \
            flex \
            git \
            libtool \
            lsb-release \
            make \
            pkg-config \
            software-properties-common \
            wget \
            unzip \
    && rm -rf /var/lib/apt/lists/*

RUN     wget https://golang.org/dl/go1.14.5.linux-amd64.tar.gz -O /tmp/go.tar.gz \
    &&  tar -xzf /tmp/go.tar.gz -C /usr/local/ \
    &&  rm -rf /tmp/go.tar.gz

ENV RUSTUP_HOME=/opt/rust \
    CARGO_HOME=/opt/rust \
    PATH=/opt/rust/bin:$PATH
RUN     curl -sSfL sh.rustup.rs | sh -s -- -y
RUN     echo $CARGO_HOME \
    &&  cargo install  exa zoxide b3sum rm-improved ripgrep

# Remove rust cruft
RUN rm -v /opt/rust/bin/cargo* \
    &&  rm -v /opt/rust/bin/rust* \
    &&  rm -v /opt/rust/bin/clippy* \
    &&  rm -v /opt/rust/bin/rls

# artifacts
WORKDIR /art

RUN git clone --depth 1 --branch jq-1.5 https://github.com/stedolan/jq.git /src/jq \
    &&  cd /src/jq \
    &&  git submodule init \
    &&  git submodule update \
    &&  autoreconf -i \
    &&  ./configure --disable-valgrind --enable-all-static --prefix=/usr/local \
    &&  make LDFLAGS=-all-static -j`nproc` \
    &&  ./jq --help \
    &&  cp jq /art/



RUN     curl -fsSL -o fzf.tar.gz https://github.com/junegunn/fzf/releases/download/0.24.3/fzf-0.24.3-linux_amd64.tar.gz \
    &&  tar xf fzf.tar.gz \
    &&  rm -rf fzf.tar.gz \
    &&  curl https://getmic.ro | bash

# Add yq to make config query work
RUN curl -sL https://github.com/mikefarah/yq/releases/download/3.4.1/yq_linux_amd64 \
    -o /art/yq &&\
    chmod +x /art/yq

FROM ${FINAL_BASE}

WORKDIR /usr/local/bin/

COPY --from=base /art/* ./
COPY --from=base /opt/rust/bin/* ./

