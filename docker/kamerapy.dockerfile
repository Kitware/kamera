FROM python:3.10.15-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.6.1 /uv /uvx /bin/

RUN apt-get update && apt-get install -yq \
    libgdal-dev \
    python3-gdal \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    redis \
    dnsutils \
    gdal-bin

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /src/kamera

# Install dependencies before copying source so this layer caches across
# code-only changes. uv.lock is gitignored: run `uv lock` before building.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

COPY ./ /src/kamera
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV PATH="/src/kamera/.venv/bin:$PATH"

ENTRYPOINT ["bash"]
