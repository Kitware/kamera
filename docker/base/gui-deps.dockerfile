## GUI deps layered on the Noetic core-deps chain.
FROM kamera/base/core-deps:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        python3-gdal \
        python3-tk \
        python3-wxgtk4.0 \
        libgl1-mesa-glx \
        libqt5x11extras5 \
        locales \
    && rm -rf /var/lib/apt/lists/*

# wxPython's wx.App init sets the en_US locale at startup; generate it so the
# GUI doesn't fail with "locale en_US cannot be set".
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

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
        psutil \
        simplekml

COPY src/core/roskv /src/roskv
RUN pip install --no-cache-dir '/src/roskv[redis]'
