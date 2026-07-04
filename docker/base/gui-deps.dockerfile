## GUI deps layered on the Jazzy core-deps chain.
FROM kamera/base/core-deps:latest

# libgl1-mesa-glx was dropped in Ubuntu 24.04; libgl1 + libglx-mesa0 replace it
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        python3-gdal \
        python3-tk \
        python3-wxgtk4.0 \
        libgl1 \
        libglx-mesa0 \
        libqt5x11extras5 \
        locales \
    && rm -rf /var/lib/apt/lists/*

# wxPython's wx.App init sets the en_US locale at startup; generate it so the
# GUI doesn't fail with "locale en_US cannot be set".
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# numpy/scipy/shapely/pyshp/pygeodesy come from core-deps; only GUI-unique deps
# here. The legacy 'PyGeodesy<19.12' pin is gone: the GUI uses
# pygeodesy.geoids.GeoidPGM, which the unpinned core-deps install provides
# (shapefile_monitor already runs against it).
RUN pip install --break-system-packages --no-cache-dir \
        Pillow \
        exifread \
        ipython \
        psutil \
        simplekml

# roskv is no longer pip-installable (its setup.py was removed in the ROS2
# port); it is built into the colcon workspace by gui.dockerfile via
# --packages-up-to wxpython_gui, which pulls in roskv as a dependency.
