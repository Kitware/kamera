## GUI deps on the Noetic core-deps chain (parallel to Kinetic core-gui-deps).
FROM kamera/base/core-deps:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        python3-gdal \
        python3-tk \
        python3-wxgtk4.0 \
        libgl1-mesa-glx \
        libqt5x11extras5 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade \
        pip \
        Pillow

# numpy/scipy/shapely/pyshp come from core-deps; only GUI-unique deps here.
# TODO: validate the unpinned pygeodesy already installed by core-deps and drop
# this <19.12 downgrade (legacy Py2-era pin, untested on Py3).
RUN pip install --no-cache-dir \
        'PyGeodesy<19.12' \
        exifread \
        ipython \
        simplekml

COPY src/core/roskv /src/roskv
RUN pip install --no-cache-dir '/src/roskv[redis]'
