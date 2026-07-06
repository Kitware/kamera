FROM debian:bookworm-slim

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -yq \
    curl \
    bzip2 \
    ca-certificates \
    make \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    redis \
    dnsutils \
 && rm -rf /var/lib/apt/lists/*

# Install micromamba
ARG MAMBA_VERSION=2.3.3
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION} \
  | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda

# The conda env supplies python + GDAL + uv; uv layers everything else into
# .venv on top of it (same flow as the native setup_postproc scripts).
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n kamera -f /tmp/environment.yml \
 && micromamba clean --all -y

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY ./ /src/kamera
WORKDIR /src/kamera

RUN eval "$(micromamba shell hook --shell bash)" \
 && micromamba activate kamera \
 && make install

ENV PATH="/src/kamera/.venv/bin:/opt/conda/envs/kamera/bin:$PATH"

ENTRYPOINT ["bash"]
