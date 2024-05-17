## this is currently ONLY supported on kinetic!
FROM kamera/kamera/kamera-deps-kinetic:latest

RUN     apt-get update && apt-get install -y \
            gdal-bin \
            python-gdal \
            python-tk \
            libgl1-mesa-glx \
            libqt5x11extras5 \
            openssh-server \
    &&  rm -rf /var/lib/apt/lists/*

RUN     pip install --upgrade \
            pip \
            Pillow


RUN     pip install --no-cache-dir \
            'PyGeodesy<19.12' \
            'IPython==5.0' \
            exifread \
            numpy \
            shapely \
            pyshp \
            scipy\
            simplekml \
            wxPython \
            typing

RUN pip install 'src/core/roskv[redis]'
