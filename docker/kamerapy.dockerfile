FROM python:3.10.15-bookworm

RUN apt-get update && apt-get install -yq \
    libgdal-dev \
    python3-gdal \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    redis \
    gdal-bin

RUN pip install --upgrade pip
RUN pip install setuptools==57.0.0

COPY ./ /src/kamera
WORKDIR /src/kamera
RUN pip install .

ENTRYPOINT ["bash"]
